import torch
import matplotlib.pyplot as plt
import numpy as np
import math

CS = {
    'FFN': 1.4,
    'Instant-NGP': 1.0,
    'GA-Planes': 2.0,
    'Grid': 4.5,
}
TEST_IDX = 10

def max_pairwise_l2_distance(arrays):
    """Find maximum pairwise L2 distance between arrays."""
    m = len(arrays)
    
    if m < 2:
        raise ValueError("Need at least 2 arrays")
    
    max_dist = 0
    estimated_eta = 0 
    
    for i in range(m):
        for j in range(i + 1, m):
            dist = np.linalg.norm(arrays[i] - arrays[j])
            if dist > max_dist:
                max_dist = dist
                estimated_eta = arrays[i] - arrays[j]
    
    return max_dist, estimated_eta

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def normalize(im):
  return (im - im.min()) / (im.max() - im.min())

def psnr_from_mse(x_pred, x_gt):
    mse = torch.mean((x_pred.flatten() - x_gt.flatten()) ** 2)
    data_range = torch.max(x_gt) - torch.min(x_gt)
    return 20*torch.log10(data_range) - 10*torch.log10(mse)

def plot_training_losses(outputs):
  plt.figure(figsize=(10,6))
  for key, output in outputs.items():
    plt.plot(output['xs'], output['losses'], label=key)
  plt.title('Training Loss')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.legend()
  plt.yscale('log')
  plt.grid(True)

def plot_error_for_convexity(gt, outputs, params, param, iters, seed=0, model_name="model"):
  flattened_gt = gt.flatten()
  optimal_errors = []
  for i, (key, output) in enumerate(outputs.items()):
    optimal_errors.append(np.linalg.norm(flattened_gt - output['best_pred'].flatten()))
  plt.figure(figsize=(10,8))
  plt.plot(params, optimal_errors)
  plt.xlabel(param)
  plt.title(f'Converged error after {iters} iterations')
  plt.ylabel("Error")
  plt.savefig(f"plots/{model_name}/{model_name}_{param}_{seed}.png")

def plot_error_upper_bound_without_optimization_gap(gt, outputs, d_star, params, param, b=5, save_tag="default", seed=0):
  assert b <= len(params), f"b={b} must be less than or equal to number of params={len(params)}"
  flattened_gt = gt.flatten()
  m = len(params)

  # Compute optimal errors
  optimal_errors = []
  for i in range(m - b):
    out = outputs[str(params[i])]
    optimal_errors.append(np.linalg.norm(flattened_gt - out['best_pred'].flatten()))
   
  # Compute upper bound without optimization gap 
  bounds_no_eta = []
  for i in range(m - b):
    out = outputs[str(params[i])]
    next_out = outputs[str(params[i + 1])]
    d = params[i]
    bound_no_eta_i = (d_star - d) * np.linalg.norm(out['best_pred'] - next_out['best_pred'])
    bounds_no_eta.append(bound_no_eta_i)

  # Compute upper bound without optimization gap block-based
  bounds_no_eta_block = []
  for i in range(m - b):
    outs = [outputs[str(params[j])] for j in range(i, i + b + 1)]
    d = params[i]
    first_term = [np.linalg.norm(outs[j]['best_pred'] - outs[j + 1]['best_pred']) for j in range(b)]
    first_term = (d_star - d) * sum(first_term) / len(first_term)
    bounds_no_eta_block.append(first_term)

  fig, ax1 = plt.subplots(figsize=(10, 8)) #
  # ax[0].figure(figsize=(10,8))
  ax2 = ax1.twinx()
  ax1.plot(params[:-b], optimal_errors, marker="o", label="Optimal Error", color="red")
  ax2.plot(params[:-b], bounds_no_eta, marker="o", label="Upper Bound without Optimization Gap")
  ax2.plot(params[:-b], bounds_no_eta_block, marker="o", label="Upper Bound without Optimization Gap (Block-based)")
  plt.xlabel(param)
  plt.ylabel("Error")
  plt.legend()
  # plt.savefig(f"./results_plots/{param}_{error_mode}_{save_tag}_{seed}.png")


def plot_error_heatmaps(gt, outputs, zoom_dim=None, model_name="FFN"):
    small_param, large_param = list(outputs.keys())
    small_pred, large_pred   = outputs[small_param]['best_pred'], outputs[large_param]['best_pred']

    small_pred = small_pred.reshape(gt.shape)
    large_pred = large_pred.reshape(gt.shape)

    model_difference = np.abs(small_pred - large_pred) * CS[model_name]
    actual_error     = np.abs(small_pred - gt)           

    if zoom_dim:
        gt_crop          = gt[zoom_dim]
        small_pred_crop  = small_pred[zoom_dim]
        large_pred_crop = large_pred[zoom_dim]
        model_difference = model_difference[zoom_dim]
        actual_error     = actual_error[zoom_dim]
    else:
        gt_crop         = gt
        small_pred_crop = small_pred
        large_pred_crop = large_pred
        
    vmin = min(model_difference.min(), actual_error.min())
    vmax = max(model_difference.max(), actual_error.max())

    fig = plt.figure(figsize=(21, 4.5))
    gs  = fig.add_gridspec(1, 5, wspace=0.01)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    axes[0].imshow(gt_crop, cmap='gray', interpolation='nearest')
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    axes[1].imshow(small_pred_crop, cmap='gray', interpolation='nearest')
    axes[1].set_title('Prediction from small model')
    axes[1].axis('off')

    axes[2].imshow(large_pred_crop, cmap='gray', interpolation='nearest')
    axes[2].set_title('Prediction from large model')
    axes[2].axis('off')

    heatmap1 = axes[3].imshow(model_difference, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[3].set_title('|small_pred − large_pred|')
    axes[3].axis('off')

    heatmap2 = axes[4].imshow(actual_error, cmap='viridis', interpolation='nearest', vmin=vmin, vmax=vmax)
    axes[4].set_title('Actual Error  |small_pred − ground_truth|')
    axes[4].axis('off')
    
    # fig.colorbar(heatmap2, ax=axes[4], fraction=0.04, pad=0.02)
    fig.text(0.11, 0.5, model_name, va='center', ha='center',
                 rotation='vertical', fontsize=14)
    fig.tight_layout(rect=[0, 0, 0.97, 1])
    plt.draw()

    pos = axes[4].get_position()           
    gap = 0.005                            
    cbar_width = 0.008                     
    cax = fig.add_axes([pos.x1 + gap, pos.y0, cbar_width, pos.height])
    fig.colorbar(heatmap2, cax=cax)
    plt.show()

def compute_model_size(model_name, model_args, n_dims=2):
    if model_name == "instant-ngp_eta":
      model_config, mlp_config = model_args['model_config'], model_args['mlp_config']
      l, w = mlp_config
      L, T, N_0, b = model_config
      F = 2
      num_mlp_params = (l - 1) * (w**2) + (L * F + l + 1) * w + 1
      num_hash_params = 0
      for l in range(L):
          N_l = math.floor(N_0 * (b**l))
          num_hash_params += min(2**T, (N_l)**n_dims) 
      # print(f'MLP params: {num_mlp_params}, Hash params: {F * num_hash_params}')
      return num_mlp_params + F * num_hash_params
    elif model_name == "ffn_eta":
      p, l, w = model_args['mapping_size'], model_args['num_layers'], model_args['width']
      return (l - 1) * (w**2) + (2*p + l + 1) * w + 1
    elif model_name == "ga-planes_eta":
      if n_dims == 2:
        r1, r2 = model_args['lineres'], model_args['planeres']
        k1, k2 = model_args['line_feature_dim'], model_args['plane_feature_dim']
        return (2 * r1 * k1) + (r2 * r2 * k2) + k1 + k2 + 1
      elif n_dims == 3:
        r1, r2, r3 = model_args['lineres'], model_args['planeres'], model_args['volumeres']
        k1, k2, k3 = model_args['line_feature_dim'], model_args['plane_feature_dim'], model_args['volume_feature_dim']
        assert k1 == k2, "For multiplication, line and plane feature dimensions must be the same"
        return (3 * r1 * k1) + (3 * r2 * r2 * k2) + (r3 * r3 * r3 * k3) + k1 + k2 + k2 + k2 + k3 + 1
      else:
        raise ValueError(f"Input dimension {n_dims} not supported for GA-Planes") 
    elif model_name == "grid_eta":
      return (model_args['grid_reso']) ** n_dims 
    else:
      raise ValueError(f"Model name {model_name} not recognized") 

def compute_params_from_model_size(model_name, model_args, model_size, n_dims=2):
    if model_name == "grid_eta":
      return math.floor(model_size ** (1/n_dims))
    elif model_name == "ffn_eta":
      p, l, w = model_args['mapping_size'], model_args['num_layers'], model_args['width']
      if p == -1:
        numerator = model_size - (l - 1) * w**2 - (l + 1) * w - 1
        return math.floor(numerator / (2 * w))
      elif w == -1:
        if l == 1:
            return math.floor((model_size - 1) / (2 * p + 2))
        else:
            a    = l - 1
            b_c  = 2 * p + l + 1
            c    = 1 - model_size
            disc = b_c**2 - 4 * a * c
            return math.floor((-b_c + math.sqrt(disc)) / (2 * a))
    elif model_name =="instant-ngp_eta":
      model_config, mlp_config = model_args['model_config'], model_args['mlp_config']
      mlp_l, w = mlp_config
      L, T, N_0, b = model_config
      F = 2
      def _hash_params(L_val, T_val):
          total = 0
          for i in range(L_val):
              N_i = math.floor(N_0 * (b ** i))
              total += min(2 ** T_val, int(N_i) ** n_dims)
          return total

      def _total(L_val, T_val):
          mlp = (mlp_l - 1) * w**2 + (L_val * F + mlp_l + 1) * w + 1
          return mlp + F * _hash_params(L_val, T_val)
      if L == -1:
        best_L, L_val = 0, 1
        while True:
            try:
                if _total(L_val, T) > model_size:
                    break
                best_L = L_val
                L_val += 1
            except (OverflowError, ValueError):
                break
        return best_L
      elif T == -1:
        mlp = (mlp_l - 1) * w**2 + (L * F + mlp_l + 1) * w + 1
        budget = model_size - mlp
        lo, hi, best_T = 0, 30, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if F * _hash_params(L, mid) <= budget:
                best_T = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best_T
    elif model_name == "ga-planes_eta":
      if n_dims == 2:
        r1, r2 = model_args['lineres'], model_args['planeres']
        k1, k2 = model_args['line_feature_dim'], model_args['plane_feature_dim']
        if r2 == -1:
          remainder = model_size - 2 * r1 * k1 - k1 - k2 - 1
          r2_val = math.floor(math.sqrt(remainder / k2))
          while (r2_val + 1)**2 * k2 <= remainder:
              r2_val += 1
          return r2_val
        elif k1 == -1:
          numerator = model_size - r2**2 * k2 - k2 - 1
          return math.floor(numerator / (2 * r1 + 1))
      elif n_dims == 3:
        r1, r2, r3 = model_args['lineres'], model_args['planeres'], model_args['volumeres']
        k1, k2, k3 = model_args['line_feature_dim'], model_args['plane_feature_dim'], model_args['volume_feature_dim']
        assert k1 == k2, "For multiplication, line and plane feature dimensions must be the same"
        if r3 == -1:
          remainder = (model_size
                             - 3 * r1 * k1 - 3 * r2**2 * k2
                             - k1 - 3 * k2 - k3 - 1)
          r3_val = math.floor((remainder / k3) ** (1 / 3))
          while (r3_val + 1)**3 * k3 <= remainder:
              r3_val += 1
          return r3_val
        elif k1 == -1 and k2 == -1:
          numerator = model_size - k3 * (r3**3 + 1) - 1
          denom     = 3 * r1 + 3 * r2**2 + 4
          return math.floor(numerator / denom)
      else:
        raise ValueError(f"Input dimension {n_dims} not supported for GA-Planes")
    else:
      raise ValueError(f"Model name {model_name} not recognized") 
      

#@title NP Area Resize Code

#from https://gist.github.com/shoyer/c0f1ddf409667650a076c058f9a17276

def _reflect_breaks(size: int) -> np.ndarray:
  """Calculate cell boundaries with reflecting boundary conditions."""
  result = np.concatenate([[0], 0.5 + np.arange(size - 1), [size - 1]])
  assert len(result) == size + 1
  return result
  
def _interval_overlap(first_breaks: np.ndarray,
                      second_breaks: np.ndarray) -> np.ndarray:
  """Return the overlap distance between all pairs of intervals.

  Args:
    first_breaks: breaks between entries in the first set of intervals, with
      shape (N+1,). Must be a non-decreasing sequence.
    second_breaks: breaks between entries in the second set of intervals, with
      shape (M+1,). Must be a non-decreasing sequence.

  Returns:
    Array with shape (N, M) giving the size of the overlapping region between
    each pair of intervals.
  """
  first_upper = first_breaks[1:]
  second_upper = second_breaks[1:]
  upper = np.minimum(first_upper[:, np.newaxis], second_upper[np.newaxis, :])

  first_lower = first_breaks[:-1]
  second_lower = second_breaks[:-1]
  lower = np.maximum(first_lower[:, np.newaxis], second_lower[np.newaxis, :])

  return np.maximum(upper - lower, 0)

def _resize_weights(
    old_size: int, new_size: int, reflect: bool = False) -> np.ndarray:
  """Create a weight matrix for resizing with the local mean along an axis.

  Args:
    old_size: old size.
    new_size: new size.
    reflect: whether or not there are reflecting boundary conditions.

  Returns:
    NumPy array with shape (new_size, old_size). Rows sum to 1.
  """
  if not reflect:
    old_breaks = np.linspace(0, old_size, num=old_size + 1)
    new_breaks = np.linspace(0, old_size, num=new_size + 1)
  else:
    old_breaks = _reflect_breaks(old_size)
    new_breaks = (old_size - 1) / (new_size - 1) * _reflect_breaks(new_size)

  weights = _interval_overlap(new_breaks, old_breaks)
  weights /= np.sum(weights, axis=1, keepdims=True)
  assert weights.shape == (new_size, old_size)
  return weights

def resize(array: np.ndarray,
           shape: [int, ...],
           reflect_axes: [int] = ()) -> np.ndarray:
  """Resize an array with the local mean / bilinear scaling.

  Works for both upsampling and downsampling in a fashion equivalent to
  block_mean and zoom, but allows for resizing by non-integer multiples. Prefer
  block_mean and zoom when possible, as this implementation is probably slower.

  Args:
    array: array to resize.
    shape: shape of the resized array.
    reflect_axes: iterable of axis numbers with reflecting boundary conditions,
      mirrored over the center of the first and last cell.

  Returns:
    Array resized to shape.

  Raises:
    ValueError: if any values in reflect_axes fall outside the interval
      [-array.ndim, array.ndim).
  """
  reflect_axes_set = set()
  for axis in reflect_axes:
    if not -array.ndim <= axis < array.ndim:
      raise ValueError('invalid axis: {}'.format(axis))
    reflect_axes_set.add(axis % array.ndim)

  output = array
  for axis, (old_size, new_size) in enumerate(zip(array.shape, shape)):
    reflect = axis in reflect_axes_set
    weights = _resize_weights(old_size, new_size, reflect=reflect)
    product = np.tensordot(output, weights, [[axis], [-1]])
    output = np.moveaxis(product, -1, axis)
  return output



def get_rays(H, W, focal, c2w):
    """Generate ray origins and directions for a camera pose."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * 0.5) / focal,
                     -(j - H * 0.5) / focal,
                     -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], rays_d.shape).copy()
    return rays_o, rays_d
 
 
def volume_render(raw, z_vals):
    """Convert network output + depth samples to rgb, depth, and acc maps."""
    rgb = torch.sigmoid(raw[..., :3])
    sigma_a = torch.relu(raw[..., 3])
    dists = torch.cat([z_vals[..., 1:] - z_vals[..., :-1],
                       torch.full_like(z_vals[..., :1], 1e10)], -1)
    alpha = 1.0 - torch.exp(-sigma_a * dists)
    trans = torch.clamp(1.0 - alpha + 1e-10, max=1.0)
    trans = torch.cat([torch.ones_like(trans[..., :1]), trans[..., :-1]], -1)
    weights = alpha * torch.cumprod(trans, -1)
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    return rgb_map, depth_map, acc_map
 
 
def normalize_pts(pts, near, far):
    """Normalize 3D points from the scene bounding box to [0, 1] for tcnn hash encoding."""
    scene_range = far 
    return (pts / scene_range) * 0.5 + 0.5  