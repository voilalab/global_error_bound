import os
import sys
import argparse
import logging
import pickle
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument('--gpu', default=0, type=int)
_early_args, _ = _early_parser.parse_known_args()
os.environ.setdefault('CUDA_VISIBLE_DEVICES', str(_early_args.gpu))
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as onp

import jax
import jax.numpy as jnp
from jax import random, jit
from jax.example_libraries import optimizers



def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--param_to_run', default='feature_dim', type=str)
  parser.add_argument('--operation', default='concatenate', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  args, _ = parser.parse_known_args()
  return args



def max_pairwise_l2_distance(vectors):
    """Return the maximum pairwise L2 distance and the difference attaining it."""
    if len(vectors) < 2:
        return 0.0, None

    max_dist = -onp.inf
    max_eta = None
    arrays = [onp.asarray(v).reshape(-1) for v in vectors]
    for i in range(len(arrays)):
        for j in range(i + 1, len(arrays)):
            diff = arrays[i] - arrays[j]
            dist = float(onp.linalg.norm(diff))
            if dist > max_dist:
                max_dist = dist
                max_eta = diff
    return max_dist, max_eta


DENSITY_BIAS = -3.0
WHITE_BACKGROUND = False
MASK_OUTSIDE_BBOX = True
TEST_IDX = 10

DEFAULT_SCENE_BBOX = (
    jnp.array([-1.5, -1.5, -1.5], dtype=jnp.float32),
    jnp.array([ 1.5,  1.5,  1.5], dtype=jnp.float32),
)


def get_rays(H, W, focal, c2w):
    i, j = jnp.meshgrid(jnp.arange(W), jnp.arange(H), indexing='xy')
    dirs = jnp.stack([(i - W * 0.5) / focal,
                      -(j - H * 0.5) / focal,
                      -jnp.ones_like(i)], axis=-1)
    rays_d = jnp.sum(dirs[..., None, :] * c2w[:3, :3], axis=-1)
    rays_o = jnp.broadcast_to(c2w[:3, -1], rays_d.shape)
    return jnp.stack([rays_o, rays_d], axis=0)


get_rays = jit(get_rays, static_argnums=(0, 1, 2))


def interp_1d(feature, coord):
    res = feature.shape[1]
    idx_f = (coord + 1.0) * 0.5 * (res - 1)
    idx0_unclipped = jnp.floor(idx_f).astype(jnp.int32)
    idx0 = jnp.clip(idx0_unclipped, 0, res - 1)
    idx1 = jnp.clip(idx0_unclipped + 1, 0, res - 1)
    w = idx_f - jnp.floor(idx_f)
    f0 = feature[:, idx0]
    f1 = feature[:, idx1]
    return ((1.0 - w)[None] * f0 + w[None] * f1).T


def interp_2d(feature, ca, cb):
    res = feature.shape[1]
    a = (ca + 1.0) * 0.5 * (res - 1)
    b = (cb + 1.0) * 0.5 * (res - 1)
    a_floor = jnp.floor(a).astype(jnp.int32)
    b_floor = jnp.floor(b).astype(jnp.int32)
    a0 = jnp.clip(a_floor, 0, res - 1)
    a1 = jnp.clip(a_floor + 1, 0, res - 1)
    b0 = jnp.clip(b_floor, 0, res - 1)
    b1 = jnp.clip(b_floor + 1, 0, res - 1)
    wa = a - jnp.floor(a)
    wb = b - jnp.floor(b)

    w00 = (1 - wa) * (1 - wb)
    w01 = (1 - wa) * wb
    w10 = wa * (1 - wb)
    w11 = wa * wb
    f00 = feature[:, a0, b0]
    f01 = feature[:, a0, b1]
    f10 = feature[:, a1, b0]
    f11 = feature[:, a1, b1]
    return (w00[None] * f00 + w01[None] * f01 +
            w10[None] * f10 + w11[None] * f11).T


def interp_3d(feature, ca, cb, cc):
    res = feature.shape[1]
    a = (ca + 1.0) * 0.5 * (res - 1)
    b = (cb + 1.0) * 0.5 * (res - 1)
    c = (cc + 1.0) * 0.5 * (res - 1)

    a_floor = jnp.floor(a).astype(jnp.int32)
    b_floor = jnp.floor(b).astype(jnp.int32)
    c_floor = jnp.floor(c).astype(jnp.int32)
    a0 = jnp.clip(a_floor, 0, res - 1); a1 = jnp.clip(a_floor + 1, 0, res - 1)
    b0 = jnp.clip(b_floor, 0, res - 1); b1 = jnp.clip(b_floor + 1, 0, res - 1)
    c0 = jnp.clip(c_floor, 0, res - 1); c1 = jnp.clip(c_floor + 1, 0, res - 1)

    wa = a - jnp.floor(a)
    wb = b - jnp.floor(b)
    wc = c - jnp.floor(c)

    f000 = feature[:, a0, b0, c0]
    f001 = feature[:, a0, b0, c1]
    f010 = feature[:, a0, b1, c0]
    f011 = feature[:, a0, b1, c1]
    f100 = feature[:, a1, b0, c0]
    f101 = feature[:, a1, b0, c1]
    f110 = feature[:, a1, b1, c0]
    f111 = feature[:, a1, b1, c1]

    w000 = (1 - wa) * (1 - wb) * (1 - wc)
    w001 = (1 - wa) * (1 - wb) * wc
    w010 = (1 - wa) * wb * (1 - wc)
    w011 = (1 - wa) * wb * wc
    w100 = wa * (1 - wb) * (1 - wc)
    w101 = wa * (1 - wb) * wc
    w110 = wa * wb * (1 - wc)
    w111 = wa * wb * wc

    return (w000[None] * f000 + w001[None] * f001 +
            w010[None] * f010 + w011[None] * f011 +
            w100[None] * f100 + w101[None] * f101 +
            w110[None] * f110 + w111[None] * f111).T


def init_gaplanes(key, r1, r2, r3, d1, d2, d3, multires_factors, hidden_dim, output_dim):
    if d1 != d2:
        raise ValueError(
            'The JAX concatenate GA-Planes implementation requires '
            f'line_feature_dim == plane_feature_dim, got {d1} and {d2}.'
        )

    K = len(multires_factors)
    decoder_in = K * 4 * d1 + d3
    n_keys = 3 * K + 3 * K + 1 + 4
    keys = random.split(key, n_keys)
    ki = 0
    params = {}

    for k, mk in enumerate(multires_factors):
        res = int(mk * r1)
        params[f'line_x_{k}'] = random.normal(keys[ki], (d1, res)) * 0.1;  ki += 1
        params[f'line_y_{k}'] = random.normal(keys[ki], (d1, res)) * 0.1;  ki += 1
        params[f'line_z_{k}'] = random.normal(keys[ki], (d1, res)) * 0.1;  ki += 1

    for k, mk in enumerate(multires_factors):
        res = int(mk * r2)
        params[f'plane_xy_{k}'] = random.normal(keys[ki], (d2, res, res)) * 0.01;  ki += 1
        params[f'plane_yz_{k}'] = random.normal(keys[ki], (d2, res, res)) * 0.01;  ki += 1
        params[f'plane_zx_{k}'] = random.normal(keys[ki], (d2, res, res)) * 0.01;  ki += 1

    params['volume'] = random.normal(keys[ki], (d3, r3, r3, r3)) * 0.001;  ki += 1

    params['W1'] = random.normal(keys[ki], (decoder_in, hidden_dim)) * 0.01;  ki += 1
    params['b1'] = jnp.zeros(hidden_dim, dtype=jnp.float32);  ki += 1
    params['W2'] = random.normal(keys[ki], (hidden_dim, output_dim)) * 0.01;  ki += 1
    params['b2'] = jnp.zeros(output_dim, dtype=jnp.float32)
    return params


def count_params(params):
    return int(sum(v.size for v in jax.tree_util.tree_leaves(params)))


def gaplanes_apply(params, pts, scene_bbox, n_levels) :
    bbox_min, bbox_max = scene_bbox
    pts_n = 2.0 * (pts - bbox_min) / (bbox_max - bbox_min) - 1.0
    pts_n = jnp.clip(pts_n, -1.0, 1.0)

    x, y, z = pts_n[:, 0], pts_n[:, 1], pts_n[:, 2]

    level_features = []
    for k in range(n_levels):
        feat_x = interp_1d(params[f'line_x_{k}'], x)
        feat_y = interp_1d(params[f'line_y_{k}'], y)
        feat_z = interp_1d(params[f'line_z_{k}'], z)

        feat_xy = interp_2d(params[f'plane_xy_{k}'], x, y)
        feat_yz = interp_2d(params[f'plane_yz_{k}'], y, z)
        feat_zx = interp_2d(params[f'plane_zx_{k}'], z, x)

        combined_lines = feat_x * feat_y * feat_z
        gated_xy = feat_xy * feat_z
        gated_yz = feat_yz * feat_x
        gated_zx = feat_zx * feat_y

        level_features.append(
            jnp.concatenate([combined_lines, gated_xy, gated_yz, gated_zx], axis=-1)
        )

    feat_vol = interp_3d(params['volume'], x, y, z)
    features = jnp.concatenate(level_features + [feat_vol], axis=-1)

    h = jax.nn.relu(features @ params['W1'] + params['b1'])
    out = h @ params['W2'] + params['b2']
    return out


def render_rays(params, key, rays, scene_bbox, n_levels, rand, allret, near, far, N_samples):
    rays_o, rays_d = rays
    bbox_min, bbox_max = scene_bbox

    z_vals = jnp.linspace(near, far, N_samples, dtype=jnp.float32)
    if rand:
        z_vals = z_vals + random.uniform(
            key, shape=list(rays_o.shape[:-1]) + [N_samples]
        ) * (far - near) / N_samples

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts_flat = jnp.reshape(pts, [-1, 3])

    if MASK_OUTSIDE_BBOX:
        inside_flat = jnp.all((pts_flat >= bbox_min) & (pts_flat <= bbox_max), axis=-1)
    else:
        inside_flat = jnp.ones((pts_flat.shape[0],), dtype=bool)

    raw = gaplanes_apply(params, pts_flat, scene_bbox, n_levels)
    raw = jnp.reshape(raw, list(pts.shape[:-1]) + [4])
    inside = jnp.reshape(inside_flat, pts.shape[:-1])

    rgb = jax.nn.sigmoid(raw[..., :3])
    sigma_a = jax.nn.softplus(raw[..., 3] + DENSITY_BIAS)
    sigma_a = sigma_a * inside

    dists = jnp.concatenate([
        z_vals[..., 1:] - z_vals[..., :-1],
        z_vals[..., -1:] - z_vals[..., -2:-1],
    ], axis=-1)
    dists = dists * jnp.linalg.norm(rays_d[..., None, :], axis=-1)

    alpha = 1.0 - jnp.exp(-sigma_a * dists)
    trans = jnp.concatenate([
        jnp.ones_like(alpha[..., :1]),
        1.0 - alpha + 1e-10,
    ], axis=-1)
    trans = jnp.cumprod(trans, axis=-1)[..., :-1]
    weights = alpha * trans

    rgb_map = jnp.sum(weights[..., None] * rgb, axis=-2)
    acc_map = jnp.sum(weights, axis=-1)

    if WHITE_BACKGROUND:
        rgb_map = rgb_map + (1.0 - acc_map)[..., None]

    if not allret:
        return rgb_map

    depth_map = jnp.sum(weights * z_vals, axis=-1)
    return rgb_map, depth_map, acc_map


render_rays_jit = jit(render_rays, static_argnums=(4, 5, 6, 9))


def render_fn(params, key, rays, scene_bbox, n_levels, rand, near, far, N_samples, row_chunk = 5):
    rets = None
    for i in range(0, rays.shape[1], row_chunk):
        out = render_rays_jit(params, key, rays[:, i:i + row_chunk],
                              scene_bbox, n_levels, rand, True,
                              near, far, N_samples)
        if rets is None:
            rets = out
        else:
            rets = tuple(jnp.concatenate([a, b], axis=0) for a, b in zip(rets, out))
    return rets


def loss_fn(params, key, rays, target, scene_bbox, n_levels, stratified, near, far, N_samples):
    rgb = render_rays_jit(params, key, rays, scene_bbox, n_levels,
                          stratified, False, near, far, N_samples)
    return jnp.mean(jnp.square(rgb - target))


def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum([jnp.sum(x * x) for x in leaves]))



MODEL_NAME = 'ga-planes_eta_small'

data = onp.load('lego_400.npz')
images_np = data['images'][..., :3].astype(onp.float32)
poses_np = data['poses'].astype(onp.float32)
focal = float(data['focal'])
H, W = images_np.shape[1:3]

images_np, val_images_np, test_images_np = onp.split(images_np, [100, 107], axis=0)
poses_np, val_poses_np, test_poses_np = onp.split(poses_np, [100, 107], axis=0)

images = jnp.asarray(images_np)
val_images = jnp.asarray(val_images_np)
test_images = jnp.asarray(test_images_np)
poses = jnp.asarray(poses_np)
val_poses = jnp.asarray(val_poses_np)
test_poses = jnp.asarray(test_poses_np)


def build_training_data(seed = 0) :
    training_rays = jnp.stack([get_rays(H, W, focal, pose) for pose in poses], axis=1)
    training_data = jnp.concatenate([training_rays, images[None]], axis=0)
    training_data = jnp.moveaxis(training_data, 0, -2)
    training_data = onp.asarray(jnp.reshape(training_data, [-1, 3, 3])).copy()
    rng = onp.random.default_rng(seed)
    rng.shuffle(training_data, axis=0)
    return training_data.astype(onp.float32, copy=False)


TRAINING_DATA = build_training_data(seed=0)



def fit_gaplanes_nerf_jax(model_params, images_arg, poses_arg, focal_arg, H_arg, W_arg, val_images, val_poses,
                          iters, learning_rate, N_samples, batch_size, near, far, device = None, seed = 0,
                          log_interval = 1000, stratified_sampling = True, fg_fraction = 0.5, fg_threshold = 0.03):
    del images_arg, poses_arg, focal_arg, H_arg, W_arg, device 

    if model_params.get('operation', 'concatenate') != 'concatenate':
        raise ValueError(
            'ga-planes_nerf_jax.ipynb implements the concatenate GA-Planes '
            f'operation only; got operation={model_params.get("operation")!r}.'
        )
    if int(model_params.get('num_decoder_layers', 2)) != 2:
        raise ValueError('The JAX notebook implementation uses a 2-layer decoder.')

    r1 = int(model_params['lineres'])
    r2 = int(model_params.get('planeres', 0))
    r3 = int(model_params.get('volumeres', 0))
    d1 = int(model_params['line_feature_dim'])
    d2 = int(model_params.get('plane_feature_dim', 0))
    d3 = int(model_params.get('volume_feature_dim', 0))
    hidden_dim = int(model_params['hidden_dim'])
    multires_factors = tuple(int(m) for m in model_params.get('multires_factors', [1, 2, 4]))
    n_levels = len(multires_factors)
    scene_bbox = model_params.get('scene_bbox', DEFAULT_SCENE_BBOX)

    rng = random.PRNGKey(seed)
    rng, init_key = random.split(rng)
    params = init_gaplanes(init_key, r1, r2, r3, d1, d2, d3,
                           multires_factors, hidden_dim)
    model_size = count_params(params)

    opt_init, opt_update, get_params = optimizers.adam(learning_rate)
    opt_state = opt_init(params)

    @jit
    def step_fn(i, opt_state, key, rays, target):
        p = get_params(opt_state)
        loss, grads = jax.value_and_grad(loss_fn)(
            p, key, rays, target, scene_bbox, n_levels,
            stratified_sampling, near, far, N_samples
        )
        return opt_update(i, grads, opt_state), loss, tree_l2_norm(grads)

    rng_np = onp.random.default_rng(seed)
    rgb_flat = TRAINING_DATA[:, 2]
    fg_idx = onp.where(onp.mean(rgb_flat, axis=-1) > fg_threshold)[0]
    bg_idx = onp.where(onp.mean(rgb_flat, axis=-1) <= fg_threshold)[0]

    b_i = 0
    best_loss = float('inf')
    best_pred = None
    best_psnr = -float('inf')
    psnrs: List[float] = []
    xs: List[int] = []
    last_train_loss = None

    for i in range(iters + 1):
        if fg_fraction is not None and len(fg_idx) > 0 and len(bg_idx) > 0:
            n_fg = int(batch_size * fg_fraction)
            n_bg = batch_size - n_fg
            batch_idx = onp.concatenate([
                rng_np.choice(fg_idx, size=n_fg, replace=True),
                rng_np.choice(bg_idx, size=n_bg, replace=True),
            ])
            rng_np.shuffle(batch_idx)
            batch = TRAINING_DATA[batch_idx]
        else:
            batch = TRAINING_DATA[b_i:b_i + batch_size]
            b_i += batch_size
            if b_i >= TRAINING_DATA.shape[0]:
                b_i = 0

        rays = jnp.moveaxis(jnp.asarray(batch[:, :2]), 1, 0)
        target = jnp.asarray(batch[:, 2])

        rng, key = random.split(rng)
        opt_state, train_loss, grad_norm = step_fn(i, opt_state, key, rays, target)
        last_train_loss = train_loss

        if i % log_interval == 0 or i == iters:
            params_i = get_params(opt_state)
            val_losses = []
            val_preds = []

            num_vals = val_poses.shape[0] if i == iters else 1
            for v in range(num_vals):
                rays_v = get_rays(H, W, focal, val_poses[v])
                rng, key = random.split(rng)
                rgb, depth, acc = render_fn(params_i, key, rays_v, scene_bbox,
                                            n_levels, False, near, far, N_samples)
                loss_v = jnp.mean(jnp.square(rgb - val_images[v]))
                val_losses.append(loss_v)
                val_preds.append(rgb)

            mean_val_loss = float(jax.device_get(jnp.mean(jnp.stack(val_losses))))
            psnr = -10.0 * onp.log10(max(mean_val_loss, 1e-12))
            psnrs.append(float(psnr))
            xs.append(i)

            if mean_val_loss < best_loss:
                best_loss = mean_val_loss
                best_psnr = float(psnr)
                best_pred = onp.asarray(jax.device_get(val_preds[0]))

            _ = jax.device_get(train_loss)
            _ = jax.device_get(grad_norm)

    final_params = get_params(opt_state)
    if best_pred is None:
        rays_v = get_rays(H, W, focal, val_poses[0])
        rng, key = random.split(rng)
        rgb, _, _ = render_fn(final_params, key, rays_v, scene_bbox,
                              n_levels, False, near, far, N_samples)
        best_pred = onp.asarray(jax.device_get(rgb))
        best_loss = float(jax.device_get(jnp.mean(jnp.square(rgb - val_images[0]))))
        best_psnr = -10.0 * onp.log10(max(best_loss, 1e-12))

    return {
        'state': final_params,
        'best_loss': float(best_loss),
        'best_psnr': float(best_psnr),
        'best_pred': best_pred,
        'model_size': model_size,
        'psnrs': psnrs,
        'xs': xs,
        'last_train_loss': float(jax.device_get(last_train_loss)) if last_train_loss is not None else None,
    }



def compute_pred_and_error(output, model_params, near, far, device, N_samples = 512):
    del device
    scene_bbox = model_params.get('scene_bbox', DEFAULT_SCENE_BBOX)
    n_levels = len(model_params.get('multires_factors', [1, 2, 4]))
    rng = random.PRNGKey(0)
    rays = get_rays(H, W, focal, test_poses[TEST_IDX])
    rgb, depth, acc = render_fn(output['state'], rng, rays, scene_bbox,
                                n_levels, False, near, far, N_samples)
    rendered = onp.asarray(jax.device_get(rgb))
    error = float(onp.linalg.norm(test_images_np[TEST_IDX] - rendered))
    return error, rendered


PARAMS_DICT = {
    'volume_resolution': {
      'volume_resos': [23, 29, 33, 37, 40, 42, 45, 47, 49, 50, 52, 53, 55, 56, 58, 59, 60, 61, 62, 63, 64],
      'lineres': 2,
      'planeres': 2,
      'line_feature_dim': 4,
      'plane_feature_dim': 4,
      'volume_feature_dim': 1,
      'hidden_dim': 128,
      'num_decoder_layers': 2,
      'multires_factors': [1, 2, 4],
    },
    'feature_dim': {
      'feature_dims': [8, 12, 16, 24, 32, 40, 48, 56, 64],
      'lineres': 200,
      'planeres': 16,
      'volumeres': 8,
      'volume_feature_dim': 4,
      'hidden_dim': 128,
      'num_decoder_layers': 2,
      'multires_factors': [1, 2, 4],
    },
}


def _ensure_output_dirs():
    os.makedirs(f'../logs/3d_nerf', exist_ok=True)
    os.makedirs(f'../results/3d_nerf', exist_ok=True)


def run_exp(args):
  _ensure_output_dirs()

  tol = 1e-6
  learning_rate, iters = 5e-4, 25000
  N_samples = 512
  batch_size = 2**10
  near, far = 2., 6.

  device = f'jax:{args.gpu}'

  if args.param_to_run == 'volume_resolution':
    logging.basicConfig(
      filename=f"logs/{MODEL_NAME}/{args.param_to_run}_s{args.seed}.log",
      level=logging.INFO,
      format='%(message)s',
      filemode='w',
      force=True,
    )
    logger = logging.getLogger(__name__)

    gap_args = {
      "lineres": PARAMS_DICT[args.param_to_run]['lineres'],
      "planeres": PARAMS_DICT[args.param_to_run]['planeres'],
      "line_feature_dim": PARAMS_DICT[args.param_to_run]['line_feature_dim'],
      "plane_feature_dim": PARAMS_DICT[args.param_to_run]['plane_feature_dim'],
      "volume_feature_dim": PARAMS_DICT[args.param_to_run]['volume_feature_dim'],
      "hidden_dim": PARAMS_DICT[args.param_to_run]['hidden_dim'],
      "operation": args.operation,
      "num_decoder_layers": PARAMS_DICT[args.param_to_run]['num_decoder_layers'],
      "multires_factors": PARAMS_DICT[args.param_to_run]['multires_factors'],
      "scene_bbox": DEFAULT_SCENE_BBOX,
    }

    volume_resos, exp_dict = PARAMS_DICT[args.param_to_run]['volume_resos'], {}
    for volume_reso in volume_resos:
      print(f"Begin volume_reso={volume_reso}")
      logger.info(f"Volume resolution={volume_reso}")
      gap_args['volumeres'] = volume_reso
      max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
      eta, list_of_outputs, i = None, [], 0
      while abs(max_dist - prev_max_dist) > tol:
          output = fit_gaplanes_nerf_jax(gap_args, images, poses, focal, H, W,
                           val_images=val_images, val_poses=val_poses,
                           iters=iters, learning_rate=learning_rate, N_samples=N_samples, batch_size=batch_size,
                           near=near, far=far, device=device, seed=args.seed + i * volume_reso)
          error, pred = compute_pred_and_error(output, gap_args, near, far, device, N_samples)
          list_of_outputs.append(output['best_pred'].flatten())
          if len(list_of_outputs) >= 2:
              prev_max_dist = max_dist
              max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
          if error < min_error:
              min_error, best_pred = error, output['best_pred'].flatten()
          logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
          i += 1
      logger.info(f"Final eta for volume resolution {volume_reso} is {max_dist}")
      exp_dict[f'{volume_reso}'] = {}
      exp_dict[f'{volume_reso}']['eta'] = eta
      exp_dict[f'{volume_reso}']['eta_upper_bound'] = max_dist
      exp_dict[f'{volume_reso}']['error'] = min_error
      exp_dict[f'{volume_reso}']['pred'] = best_pred
      exp_dict[f'{volume_reso}']['model_size'] = output['model_size']
      logger.info("-----------------------------------------------------------")
      print(f"Finish volume_reso={volume_reso}")

    with open(f"pkls/{MODEL_NAME}/{args.param_to_run}_s{args.seed}.pkl", "wb") as f:
        pickle.dump(exp_dict, f)

  elif args.param_to_run == 'feature_dim':
    logging.basicConfig(
      filename=f"../logs/3d_nerf/{MODEL_NAME}.log",
      level=logging.INFO,
      format='%(message)s',
      filemode='w',
      force=True,
    )
    logger = logging.getLogger(__name__)

    gap_args = {
      "lineres": PARAMS_DICT[args.param_to_run]['lineres'],
      "planeres": PARAMS_DICT[args.param_to_run]['planeres'],
      "volumeres": PARAMS_DICT[args.param_to_run]['volumeres'],
      "volume_feature_dim": PARAMS_DICT[args.param_to_run]['volume_feature_dim'],
      "hidden_dim": PARAMS_DICT[args.param_to_run]['hidden_dim'],
      "operation": args.operation,
      "num_decoder_layers": PARAMS_DICT[args.param_to_run]['num_decoder_layers'],
      "multires_factors": PARAMS_DICT[args.param_to_run]['multires_factors'],
      "scene_bbox": DEFAULT_SCENE_BBOX,
    }
    pkl_path = f"../results/3d_nerf/{MODEL_NAME}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({}, f)

    feature_dims, exp_dict = PARAMS_DICT[args.param_to_run]['feature_dims'], {}
    for feature_dim in feature_dims:
      print(f"Begin feature_dim={feature_dim}")
      logger.info(f"Feature Dimension={feature_dim}")
      gap_args['line_feature_dim'] = feature_dim
      gap_args['plane_feature_dim'] = feature_dim
      max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
      eta, list_of_outputs, i = None, [], 0
      while abs(max_dist - prev_max_dist) > tol:
          output = fit_gaplanes_nerf_jax(gap_args, images, poses, focal, H, W,
                           val_images=val_images, val_poses=val_poses,
                           iters=iters, learning_rate=learning_rate, N_samples=N_samples, batch_size=batch_size,
                           near=near, far=far, device=device, seed=args.seed + i * feature_dim, log_interval=1000)
          error, pred = compute_pred_and_error(output, gap_args, near, far, device, N_samples)
          list_of_outputs.append(pred)
          if len(list_of_outputs) >= 2:
              prev_max_dist = max_dist
              max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
          if error < min_error:
              min_error, best_pred = error, pred
          logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
          i += 1
      logger.info(f"Final eta for feature dimension {feature_dim} is {max_dist}")
      logger.info("-----------------------------------------------------------")
      print(f"Finish feature_dim={feature_dim}")

      with open(pkl_path, "rb") as f:
          exp_dict = pickle.load(f)
      exp_dict[f'{feature_dim}'] = {
          'eta': eta,
          'eta_upper_bound': max_dist,
          'error': min_error,
          'pred': best_pred,
          'model_size': output['model_size'],
      }
      with open(pkl_path, "wb") as f:
          pickle.dump(exp_dict, f)
  else:
    print(f"Parameter {args.param_to_run} not recognized. Skipping.")

  logging.shutdown()


if __name__ == "__main__":
    args = parse_args()
    run_exp(args)
