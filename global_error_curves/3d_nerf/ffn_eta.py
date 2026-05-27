import os
import sys

def _set_jax_gpu_from_argv():
    gpu = "0"
    if "--gpu" in sys.argv:
        idx = sys.argv.index("--gpu")
        if idx + 1 < len(sys.argv):
            gpu = sys.argv[idx + 1]
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu)

_set_jax_gpu_from_argv()

import argparse
import logging
import pickle
import numpy as onp

import jax
from jax import jit, random
import jax.numpy as jnp
from jax.example_libraries import optimizers, stax

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--param_to_run', default='width', type=str,
                      choices=['mapping_size', 'width', 'depth'])
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  parser.add_argument('--data_path', default='lego_400.npz', type=str)
  parser.add_argument('--test_idx', default=0, type=int)
  parser.add_argument('--iters', default=50000, type=int)
  parser.add_argument('--learning_rate', default=5e-4, type=float)
  parser.add_argument('--batch_size', default=1024, type=int)
  parser.add_argument('--n_samples', default=512, type=int)
  parser.add_argument('--log_interval', default=25000, type=int)
  parser.add_argument('--eval_chunk', default=512, type=int)
  parser.add_argument('--tol', default=1e-6, type=float)
  parser.add_argument('--near', default=2.0, type=float)
  parser.add_argument('--far', default=6.0, type=float)
  parser.add_argument('--gaussian_scale', default=8.0, type=float)
  parser.add_argument('--stratified_sampling', action='store_true', default=True)
  parser.add_argument('--no_stratified_sampling', dest='stratified_sampling', action='store_false')
  args, _ = parser.parse_known_args()
  return args


PARAMS_DICT = {
    'mapping_size': {
        'mapping_sizes': [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256,
                          320, 384, 448, 512],
        'num_layers': 4,
        'width': 512,
        'threshold': 0,
    },
    'width': {
        'widths': [64, 96, 128, 192, 256, 384, 512, 1024],
        'mapping_size': 512,
        'num_layers': 4,
        'threshold': 0,
    },
    'depth': {
        'nums_layers': onp.arange(1, 8, 1),
        'mapping_size': 512,
        'width': 512,
    },
}

MODEL_NAME = 'ffn_eta'

def ensure_output_dirs():
  os.makedirs(f'../logs/3d_nerf', exist_ok=True)
  os.makedirs(f'../results/3d_nerf', exist_ok=True)


def max_pairwise_l2_distance(outputs):
    """Return the maximum pairwise L2 distance and the corresponding eta vector."""
    max_dist = -onp.inf
    eta = None
    for i in range(len(outputs)):
        a = onp.asarray(outputs[i]).reshape(-1)
        for j in range(i + 1, len(outputs)):
            b = onp.asarray(outputs[j]).reshape(-1)
            diff = a - b
            dist = float(onp.linalg.norm(diff))
            if dist > max_dist:
                max_dist = dist
                eta = diff
    return max_dist, eta


def tree_numel(tree):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(tree)))


def load_lego_data(data_path):
    data = onp.load(data_path)
    images = data['images'].astype(onp.float32)
    poses = data['poses'].astype(onp.float32)
    focal = float(data['focal'])
    H, W = images.shape[1:3]

    images, val_images, test_images = onp.split(images[..., :3], [100, 107], axis=0)
    poses, val_poses, test_poses = onp.split(poses, [100, 107], axis=0)
    return images, poses, val_images, val_poses, test_images, test_poses, focal, H, W


def get_rays_np(H, W, focal, c2w):
    i, j = onp.meshgrid(onp.arange(W), onp.arange(H), indexing='xy')
    dirs = onp.stack([(i - W * 0.5) / focal,
                      -(j - H * 0.5) / focal,
                      -onp.ones_like(i)], axis=-1).astype(onp.float32)
    rays_d = onp.sum(dirs[..., onp.newaxis, :] * c2w[:3, :3], axis=-1)
    rays_o = onp.broadcast_to(c2w[:3, -1], rays_d.shape)
    return onp.stack([rays_o, rays_d], axis=0).astype(onp.float32)


def build_training_data(images, poses, focal, H, W, seed=0):
    training_rays = onp.stack([get_rays_np(H, W, focal, pose) for pose in poses], axis=1)
    training_data = onp.concatenate([training_rays, images[None]], axis=0)
    training_data = onp.moveaxis(training_data, 0, -2)
    training_data = onp.reshape(training_data, [-1, 3, 3]).astype(onp.float32)
    rng = onp.random.default_rng(seed)
    rng.shuffle(training_data)
    return jnp.asarray(training_data)


def make_network(num_layers, num_channels):
    layers = []
    for _ in range(num_layers - 1):
        layers.append(stax.Dense(num_channels))
        layers.append(stax.Relu)
    layers.append(stax.Dense(4))
    return stax.serial(*layers)


def render_rays(apply_fn, params, avals, bvals, key, rays, near, far, N_samples,
                rand=False, allret=False):
    rays_o, rays_d = rays

    z_vals = jnp.linspace(near, far, N_samples, dtype=rays_o.dtype)
    if rand:
        z_vals = z_vals + random.uniform(
            key,
            shape=list(rays_o.shape[:-1]) + [N_samples],
            dtype=rays_o.dtype,
        ) * ((far - near) / N_samples)

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts_flat = jnp.reshape(pts, [-1, 3])

    if bvals is not None:
        features = pts_flat @ bvals.T
        pts_flat = jnp.concatenate([avals * jnp.sin(features),
                                    avals * jnp.cos(features)], axis=-1)

    raw = apply_fn(params, pts_flat)
    raw = jnp.reshape(raw, list(pts.shape[:-1]) + [4])

    rgb = jax.nn.sigmoid(raw[..., :3])
    sigma_a = jax.nn.relu(raw[..., 3])

    dists = jnp.concatenate([
        z_vals[..., 1:] - z_vals[..., :-1],
        jnp.broadcast_to(jnp.array([1e10], dtype=z_vals.dtype), z_vals[..., :1].shape),
    ], axis=-1)

    alpha = 1.0 - jnp.exp(-sigma_a * dists)
    trans = jnp.minimum(1.0, 1.0 - alpha + 1e-10)
    trans = jnp.concatenate([jnp.ones_like(trans[..., :1]), trans[..., :-1]], axis=-1)
    weights = alpha * jnp.cumprod(trans, axis=-1)

    rgb_map = jnp.sum(weights[..., None] * rgb, axis=-2)
    acc_map = jnp.sum(weights, axis=-1)

    if not allret:
        return rgb_map

    depth_map = jnp.sum(weights * z_vals, axis=-1)
    return rgb_map, depth_map, acc_map


def render_image(params, avals, bvals, num_layers, width, pose, focal, H, W,
                 near, far, N_samples, chunk=2048):
    _, apply_fn = make_network(num_layers, width)

    @jit
    def render_chunk(params, rays):
        return render_rays(apply_fn, params, avals, bvals, random.PRNGKey(0), rays,
                           near, far, N_samples, rand=False, allret=False)

    rays = get_rays_np(H, W, focal, pose)
    rays_flat = rays.reshape(2, -1, 3)
    rgb_chunks = []
    for start in range(0, rays_flat.shape[1], chunk):
        rays_c = jnp.asarray(rays_flat[:, start:start + chunk])
        rgb_c = render_chunk(params, rays_c)
        rgb_chunks.append(onp.asarray(jax.device_get(rgb_c)))
    return onp.concatenate(rgb_chunks, axis=0).reshape(H, W, 3)


def fit_fourier_features_nerf_jax(ffn_args, training_data, val_images, val_poses,
                                  focal, H, W, iters, learning_rate, N_samples,
                                  batch_size, near, far, seed=0,
                                  stratified=True, log_interval=1000,
                                  eval_chunk=2048):
    num_layers = int(ffn_args['num_layers'])
    width = int(ffn_args['width'])
    bvals = jnp.asarray(ffn_args['B'], dtype=jnp.float32)
    avals = jnp.ones((bvals.shape[0],), dtype=jnp.float32)

    init_fn, apply_fn = make_network(num_layers, width)
    init_shape = (-1, bvals.shape[0] * 2)

    rng = random.PRNGKey(seed)
    rng, init_key = random.split(rng)
    _, net_params = init_fn(init_key, init_shape)

    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(net_params)

    def render_train(params, key, rays):
        return render_rays(apply_fn, params, avals, bvals, key, rays,
                           near, far, N_samples, rand=stratified, allret=False)

    @jit
    def loss_fn(params, key, rays, target):
        rgb = render_train(params, key, rays)
        return jnp.mean(jnp.square(rgb - target))

    @jit
    def step_fn(i, opt_state, key, rays, target):
        params = get_params(opt_state)
        loss, grads = jax.value_and_grad(loss_fn)(params, key, rays, target)
        return opt_update(i, grads, opt_state), loss

    n_train = training_data.shape[0]
    b_i = 0
    best_loss = onp.inf
    best_state = None
    last_train_loss = onp.inf

    for step in range(iters + 1):
        if b_i + batch_size > n_train:
            b_i = 0
        batch = training_data[b_i:b_i + batch_size]
        b_i += batch_size

        rays = jnp.moveaxis(batch[:, :2], 1, 0)
        target = batch[:, 2]
        rng, key = random.split(rng)
        opt_state, train_loss = step_fn(step, opt_state, key, rays, target)
        last_train_loss = float(jax.device_get(train_loss))

        if step % log_interval == 0 or step == iters:
            params = get_params(opt_state)
            val_losses = []
            for v in range(min(1, val_poses.shape[0])):
                pred = render_image(params, avals, bvals, num_layers, width,
                                    val_poses[v], focal, H, W, near, far,
                                    N_samples, chunk=eval_chunk)
                val_losses.append(float(onp.mean((pred - val_images[v]) ** 2)))
            val_loss = float(onp.mean(val_losses))
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = jax.device_get(params)

    if best_state is None:
        best_state = jax.device_get(get_params(opt_state))
        best_loss = last_train_loss

    trainable_params = tree_numel(best_state)
    encoding_params = int(bvals.size + avals.size)

    return {
        'state': best_state,
        'avals': onp.asarray(jax.device_get(avals)),
        'bvals': onp.asarray(jax.device_get(bvals)),
        'num_layers': num_layers,
        'width': width,
        'mapping_size': int(bvals.shape[0]),
        'best_loss': best_loss,
        'train_loss': last_train_loss,
        'trainable_params': trainable_params,
        'encoding_params': encoding_params,
        'model_size': trainable_params + encoding_params,
    }


def compute_pred_and_error(output, test_images, test_poses, focal, H, W,
                           near, far, N_samples, test_idx=0, chunk=2048):
    test_idx = int(test_idx) % test_images.shape[0]
    params = output['state']
    avals = jnp.asarray(output['avals'], dtype=jnp.float32)
    bvals = jnp.asarray(output['bvals'], dtype=jnp.float32)
    rendered = render_image(params, avals, bvals, output['num_layers'], output['width'],
                            test_poses[test_idx], focal, H, W, near, far, N_samples,
                            chunk=chunk)
    error = float(onp.linalg.norm(test_images[test_idx] - rendered))
    return error, rendered


def make_bvals(max_mapping_size, gaussian_scale, seed):
    key = random.PRNGKey(seed)
    return random.normal(key, (int(max_mapping_size), 3), dtype=jnp.float32) * gaussian_scale


def save_result_incremental(pkl_path, key, eta, max_dist, min_error, best_pred, output):
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            exp_dict = pickle.load(f)
    else:
        exp_dict = {}

    exp_dict[str(key)] = {
        'eta': eta,
        'eta_upper_bound': max_dist,
        'error': min_error,
        'pred': best_pred,
        'model_size': output['model_size'],
        'trainable_params': output['trainable_params'],
        'encoding_params': output['encoding_params'],
        'best_loss': output['best_loss'],
    }
    with open(pkl_path, 'wb') as f:
        pickle.dump(exp_dict, f)


def run_exp(args):
  ensure_output_dirs()

  tol = args.tol
  learning_rate, iters = args.learning_rate, args.iters
  N_samples = args.n_samples
  batch_size = args.batch_size
  near, far = args.near, args.far

  images, poses, val_images, val_poses, test_images, test_poses, focal, H, W = load_lego_data(args.data_path)
  training_data = build_training_data(images, poses, focal, H, W, seed=args.seed)

  logging.basicConfig(
    filename=f"../logs/3d_nerf/{MODEL_NAME}.log",
    level=logging.INFO,
    format='%(message)s',
    filemode='w',
    force=True,
  )
  logger = logging.getLogger(__name__)
  logger.info(f"JAX backend={jax.default_backend()}, devices={jax.devices()}")
  logger.info(f"Data path={args.data_path}, H={H}, W={W}, focal={focal}")
  logger.info(f"iters={iters}, lr={learning_rate}, batch_size={batch_size}, N_samples={N_samples}, near={near}, far={far}")

  pkl_path = f"../results/3d_nerf/{MODEL_NAME}.pkl"
  with open(pkl_path, 'wb') as f:
      pickle.dump({}, f)

  if args.param_to_run == 'mapping_size':
      num_layers = PARAMS_DICT[args.param_to_run]['num_layers']
      width = PARAMS_DICT[args.param_to_run]['width']
      mapping_sizes = PARAMS_DICT[args.param_to_run]['mapping_sizes']
      B_full = make_bvals(max(mapping_sizes), args.gaussian_scale, args.seed)

      for mapping_size in mapping_sizes:
          print(f"Begin mapping_size={mapping_size}")
          logger.info(f"mapping_size={mapping_size}")
          ffn_args = {
              'num_layers': num_layers,
              'width': width,
              'B': B_full[:mapping_size, :],
          }
          max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
          eta, list_of_outputs, i = None, [], 0
          while abs(max_dist - prev_max_dist) > tol:
              output = fit_fourier_features_nerf_jax(
                  ffn_args, training_data, val_images, val_poses, focal, H, W,
                  iters=iters, learning_rate=learning_rate, N_samples=N_samples,
                  batch_size=batch_size, near=near, far=far,
                  seed=args.seed + i * mapping_size,
                  log_interval=args.log_interval,
                  eval_chunk=args.eval_chunk,
              )
              error, pred = compute_pred_and_error(
                  output, test_images, test_poses, focal, H, W,
                  near=near, far=far, N_samples=N_samples,
                  test_idx=args.test_idx, chunk=args.eval_chunk,
              )
              list_of_outputs.append(pred.flatten())
              i += 1
              if mapping_size > PARAMS_DICT[args.param_to_run]['threshold'] or i > 5:
                  if len(list_of_outputs) >= 2:
                      prev_max_dist = max_dist
                      max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
                  if error < min_error:
                      min_error, best_pred = error, pred.flatten()
                  logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
              else:
                  logger.info(f"   Iteration {i} with loss={output['best_loss']:.3e} and error={error}")

          logger.info(f"Final eta for mapping size {mapping_size} is {max_dist}")
          save_result_incremental(pkl_path, mapping_size, eta, max_dist, min_error, best_pred, output)
          logger.info("-----------------------------------------------------------")
          print(f"Finish mapping_size={mapping_size}")

  elif args.param_to_run == 'width':
      num_layers = PARAMS_DICT[args.param_to_run]['num_layers']
      mapping_size = PARAMS_DICT[args.param_to_run]['mapping_size']
      widths = PARAMS_DICT[args.param_to_run]['widths']
      B = make_bvals(mapping_size, args.gaussian_scale, args.seed)

      for width in widths:
          print(f"Begin width={width}")
          logger.info(f"width={width}")
          ffn_args = {
              'num_layers': num_layers,
              'width': width,
              'B': B,
          }
          max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
          eta, list_of_outputs, i = None, [], 0
          while abs(max_dist - prev_max_dist) > tol:
              output = fit_fourier_features_nerf_jax(
                  ffn_args, training_data, val_images, val_poses, focal, H, W,
                  iters=iters, learning_rate=learning_rate, N_samples=N_samples,
                  batch_size=batch_size, near=near, far=far,
                  seed=args.seed + i * width,
                  log_interval=args.log_interval,
                  eval_chunk=args.eval_chunk,
              )
              error, pred = compute_pred_and_error(
                  output, test_images, test_poses, focal, H, W,
                  near=near, far=far, N_samples=N_samples,
                  test_idx=args.test_idx, chunk=args.eval_chunk,
              )
              list_of_outputs.append(pred.flatten())
              i += 1
              if width > PARAMS_DICT[args.param_to_run]['threshold'] or i > 5:
                  if len(list_of_outputs) >= 2:
                      prev_max_dist = max_dist
                      max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
                  if error < min_error:
                      min_error, best_pred = error, pred.flatten()
                  logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
              else:
                  logger.info(f"   Iteration {i} with loss={output['best_loss']:.3e} and error={error}")

          logger.info(f"Final eta for width {width} is {max_dist}")
          save_result_incremental(pkl_path, width, eta, max_dist, min_error, best_pred, output)
          logger.info("-----------------------------------------------------------")
          print(f"Finish width={width}")

  elif args.param_to_run == 'depth':
      width = PARAMS_DICT[args.param_to_run]['width']
      mapping_size = PARAMS_DICT[args.param_to_run]['mapping_size']
      nums_layers = PARAMS_DICT[args.param_to_run]['nums_layers']
      B = make_bvals(mapping_size, args.gaussian_scale, args.seed)

      for num_layers in nums_layers:
          num_layers = int(num_layers)
          print(f"Begin num_layers={num_layers}")
          logger.info(f"Number of hidden layers={num_layers}")
          ffn_args = {
              'num_layers': num_layers,
              'width': width,
              'B': B,
          }
          max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
          eta, list_of_outputs, i = None, [], 0
          while abs(max_dist - prev_max_dist) > tol:
              output = fit_fourier_features_nerf_jax(
                  ffn_args, training_data, val_images, val_poses, focal, H, W,
                  iters=iters, learning_rate=learning_rate, N_samples=N_samples,
                  batch_size=batch_size, near=near, far=far,
                  seed=args.seed + i * num_layers,
                  log_interval=args.log_interval,
                  eval_chunk=args.eval_chunk,
              )
              error, pred = compute_pred_and_error(
                  output, test_images, test_poses, focal, H, W,
                  near=near, far=far, N_samples=N_samples,
                  test_idx=args.test_idx, chunk=args.eval_chunk,
              )
              list_of_outputs.append(pred.flatten())
              if len(list_of_outputs) >= 2:
                  prev_max_dist = max_dist
                  max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
              if error < min_error:
                  min_error, best_pred = error, pred.flatten()
              logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
              i += 1

          logger.info(f"Final eta for number of hidden layers {num_layers} is {max_dist}")
          save_result_incremental(pkl_path, num_layers, eta, max_dist, min_error, best_pred, output)
          logger.info("-----------------------------------------------------------")
          print(f"Finish num_layers={num_layers}")

  else:
      print(f"Parameter {args.param_to_run} not recognized. Skipping.")

  logging.shutdown()


if __name__ == "__main__":
    args = parse_args()
    run_exp(args)
