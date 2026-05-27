import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import numpy as np
import math
import time
from helpers import *

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def _sinc(x: torch.Tensor) -> torch.Tensor:
    pix = math.pi * x
    return torch.where(torch.abs(x) < 1e-8, torch.ones_like(x), torch.sin(pix) / pix)

@torch.no_grad()
def _lanczos_weights_1d(in_len: int, out_len: int, a: int, device, dtype):
    scale = in_len / out_len
    j = torch.arange(out_len, device=device, dtype=dtype)
    center = (j + 0.5) * scale - 0.5

    offsets = torch.arange(-a, a + 1, device=device)
    K = offsets.numel()

    base = torch.floor(center).to(torch.int64)
    idx = base[:, None] + offsets[None, :]
    dist = center[:, None] - idx.to(dtype)

    w = _sinc(dist) * _sinc(dist / float(a))
    w = torch.where(torch.abs(dist) <= a + 1e-9, w, torch.zeros_like(w))

    mask = (idx >= 0) & (idx < in_len)
    w = w * mask.to(dtype)

    w_sum = w.sum(dim=1, keepdim=True).clamp_min(1e-12)
    w = w / w_sum

    idx_clamped = idx.clamp(min=0, max=in_len - 1)
    return idx_clamped, w


def _resample_1d_with_weights(x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, dim: int) -> torch.Tensor:
    """Resample along a single spatial dim. Works for 4D (N,C,H,W) or 5D (N,C,D,H,W)."""
    ndim = x.ndim
    assert ndim in (4, 5)
    spatial_dims = list(range(2, ndim))  # [2,3] or [2,3,4]
    assert dim in spatial_dims

    shape = x.shape
    in_len = shape[dim]
    out_len, K = w.shape

    # Move target dim to last, flatten everything else
    perm = list(range(ndim))
    perm.remove(dim)
    perm.append(dim)
    x_tr = x.permute(*perm).contiguous()
    M = x_tr[..., 0].numel()
    x_flat = x_tr.reshape(M, in_len)

    idx_flat = idx.reshape(1, -1).expand(M, -1)
    samples = torch.gather(x_flat, 1, idx_flat).view(M, out_len, K)
    y_flat = (samples * w.view(1, out_len, K)).sum(dim=2)  # (M, out_len)

    # Rebuild shape with out_len at the end, then permute back
    new_trailing_shape = list(shape[:dim]) + list(shape[dim+1:]) + [out_len]
    y = y_flat.view(new_trailing_shape)

    # Inverse permutation
    inv = [0] * ndim
    for i, p in enumerate(perm):
        inv[p] = i
    y = y.permute(*inv).contiguous()
    return y


def _fftshift_nd(x, ndim_spatial):
    dims = tuple(range(-ndim_spatial, 0))
    return torch.fft.fftshift(x, dim=dims)

def _ifftshift_nd(x, ndim_spatial):
    dims = tuple(range(-ndim_spatial, 0))
    return torch.fft.ifftshift(x, dim=dims)


def ideal_sinc_resize(x: torch.Tensor, out_N: int) -> torch.Tensor:
    """FFT-based sinc resize for square 2D or cubic 3D signals."""
    ndim = x.ndim
    assert ndim in (4, 5), "Expected 4D (N,C,H,W) or 5D (N,C,D,H,W)"
    spatial = ndim - 2  # 2 or 3
    r = x.shape[2]
    assert all(x.shape[i] == r for i in range(2, ndim)), "Spatial dims must be equal."

    if out_N == r:
        return x

    fft_fn = torch.fft.fft2 if spatial == 2 else torch.fft.fftn
    ifft_fn = torch.fft.ifft2 if spatial == 2 else torch.fft.ifftn
    fft_kwargs = {} if spatial == 2 else {"dim": (-3, -2, -1)}
    ifft_kwargs = fft_kwargs

    X = _fftshift_nd(fft_fn(x, **fft_kwargs), spatial)

    if out_N > r:
        pad = out_N - r
        pl = pad // 2
        out_shape = list(x.shape[:2]) + [out_N] * spatial
        X_pad = torch.zeros(out_shape, dtype=X.dtype, device=X.device)
        slices = [slice(None), slice(None)] + [slice(pl, pl + r)] * spatial
        X_pad[tuple(slices)] = X
        Xr = X_pad
    else:
        crop = r - out_N
        cl = crop // 2
        slices = [slice(None), slice(None)] + [slice(cl, cl + out_N)] * spatial
        Xr = X[tuple(slices)]

    y = ifft_fn(_ifftshift_nd(Xr, spatial), **ifft_kwargs)
    scale = (out_N / r) ** spatial
    return (scale * y).real


def sinc_interp_separable(x: torch.Tensor, idx_w_pairs: list) -> torch.Tensor:
    y = x
    for idx, w, dim in idx_w_pairs:
        y = _resample_1d_with_weights(y, idx, w, dim=dim)
    return y


def fit_grid(target_signal, r=128, iters=500, lr=1e-2, interp="bilinear", log_interval=500, seed=0, device=device, mask=None, a=3, count_params=False):
    torch.manual_seed(seed)
    train_signal = torch.tensor(target_signal, dtype=torch.float32, device=device)

    spatial_ndim = train_signal.ndim  # 2 or 3
    train_signal = train_signal.unsqueeze(0) 

    spatial_ndim = train_signal.ndim - 1
    C = train_signal.shape[0]
    spatial_shape = train_signal.shape[1:]

    # Precompute masked FFT target
    if mask is not None:
        fft_axes = tuple(range(-spatial_ndim, 0))
        mask_torch = torch.tensor(mask, dtype=torch.complex64, device=device)
        y_train_fft = torch.fft.fftn(train_signal.to(torch.complex64), dim=fft_axes)

    grid_shape = [1, C] + [r] * spatial_ndim
    model = torch.rand(*grid_shape, requires_grad=True, device=device)
    if count_params: print(f'Number of parameters: {r**2}')

    lanczos_pairs = []
    if interp == 'lanczos':
        for i, s in enumerate(spatial_shape):
            dim = i + 2
            idx, w = _lanczos_weights_1d(r, s, a=a, device=device, dtype=torch.float32)
            lanczos_pairs.append((idx, w, dim))

    optimizer = optim.Adam([model], lr=lr)
    losses, xs = [], []
    best_error, best_pred = float('inf'), None

    def _upsample(m):
        if interp == 'lanczos':
            return sinc_interp_separable(m, lanczos_pairs).squeeze(0)
        elif interp == 'sinc':
            return ideal_sinc_resize(m, spatial_shape[0]).squeeze(0)
        else:
            mode = interp
            if spatial_ndim == 3:
                mode = 'trilinear' if interp == 'bilinear' else interp
            return nn.functional.interpolate(
                m, size=spatial_shape, mode=mode, align_corners=False
            ).squeeze(0)

    for i in range(iters):
        optimizer.zero_grad()
        signal_hat = _upsample(model)

        if mask is not None:
            fft_axes = tuple(range(-spatial_ndim, 0))
            fft_hat = torch.fft.fftn(signal_hat.to(torch.complex64), dim=fft_axes)
            masked = fft_hat * mask_torch
            loss = 0.5 * torch.mean(torch.abs(masked - y_train_fft) ** 2)
        else:
            loss = torch.mean((signal_hat - train_signal) ** 2)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        xs.append(i)

        if loss.item() < best_error:
            best_error = loss.item()
            best_pred = signal_hat.squeeze().detach().cpu().numpy()

        if (i + 1) % log_interval == 0:
            print(f'Iteration {i+1}/{iters}, Loss: {loss.item():.3e}')

    with torch.no_grad():
        pred_signal = _upsample(model)

    return {
        'state': r,
        'pred': pred_signal.cpu().detach().numpy(),
        'losses': losses,
        'xs': xs,
        'best_pred': best_pred,
        'best_loss': best_error,
    }

def sample_voxel_grid(grid, pts, bbox_min, bbox_max):
    normalised = 2.0 * (pts - bbox_min) / (bbox_max - bbox_min) - 1.0
 
    sample_coords = normalised[:, [2, 1, 0]]          # (N,3) reorder xyz → whd for grid_sample's (x,y,z)
    sample_coords = sample_coords.view(1, 1, 1, -1, 3)  # (1,1,1,N,3)
 
    sampled = F.grid_sample(grid, sample_coords,
                            mode='bilinear', padding_mode='zeros',
                            align_corners=True)          # (1,C,1,1,N)
    return sampled.view(grid.shape[1], -1).permute(1, 0)  # (N, C)
 
 
def fit_grid_nerf(images, poses, focal, H, W,
                  val_images=None, val_poses=None,
                  r=128, iters=50000, lr=1e-2,
                  batch_size=1024, N_samples=128,
                  near=2., far=6., stratified=True,
                  bbox_min=None, bbox_max=None,
                  tv_weight=0.0,
                  log_interval=5000, seed=0, device=device,
                  count_params=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
 
    all_rays_o_list, all_rays_d_list, all_rgb_list = [], [], []
    for i in range(images.shape[0]):
        ro, rd = get_rays(H, W, focal, poses[i])
        all_rays_o_list.append(ro.reshape(-1, 3))
        all_rays_d_list.append(rd.reshape(-1, 3))
        all_rgb_list.append(images[i][..., :3].reshape(-1, 3))
    all_rays_o = np.concatenate(all_rays_o_list, 0)
    all_rays_d = np.concatenate(all_rays_d_list, 0)
    all_rgb    = np.concatenate(all_rgb_list, 0)
 
    perm = np.random.permutation(all_rays_o.shape[0])
    all_rays_o = all_rays_o[perm]
    all_rays_d = all_rays_d[perm]
    all_rgb    = all_rgb[perm]
 
    if bbox_min is None:
        bbox_min = np.array([-1.5, -1.5, -1.5], dtype=np.float32)
    if bbox_max is None:
        bbox_max = np.array([ 1.5,  1.5,  1.5], dtype=np.float32)
    bbox_min_t = torch.tensor(bbox_min, dtype=torch.float32, device=device)
    bbox_max_t = torch.tensor(bbox_max, dtype=torch.float32, device=device)
 
    grid = torch.zeros(1, 4, r, r, r, device=device, requires_grad=True)
    nn.init.uniform_(grid.data, -0.1, 0.1)
    num_params = r ** 3 * 4
    if count_params:
        print(f'Number of parameters: {num_params}')
 
    optimizer = optim.Adam([grid], lr=lr)
    losses, xs = [], []
    best_loss = float('inf')
    b_i = 0
    t0 = time.time()
 
    for it in range(iters):
        if b_i + batch_size > all_rays_o.shape[0]:
            b_i = 0
        ro = torch.tensor(all_rays_o[b_i:b_i + batch_size],
                          dtype=torch.float32, device=device)
        rd = torch.tensor(all_rays_d[b_i:b_i + batch_size],
                          dtype=torch.float32, device=device)
        target = torch.tensor(all_rgb[b_i:b_i + batch_size],
                              dtype=torch.float32, device=device)
        b_i += batch_size
 
        z_vals = torch.linspace(near, far, N_samples, device=device)
        if stratified:
            z_vals = z_vals + torch.rand(batch_size, N_samples, device=device) * (far - near) / N_samples
        else:
            z_vals = z_vals.unsqueeze(0).expand(batch_size, -1)
 
        pts = ro[:, None, :] + rd[:, None, :] * z_vals[:, :, None]
 
        raw = sample_voxel_grid(grid, pts.reshape(-1, 3),
                                bbox_min_t, bbox_max_t)   # (batch*N_samples, 4)
        raw = raw.reshape(batch_size, N_samples, 4)
 
        rgb_map, _, _ = volume_render(raw, z_vals)
 
        loss = torch.mean((rgb_map - target) ** 2)
 
        # Optional TV regularisation for smoother grids
        if tv_weight > 0:
            tv = (torch.diff(grid, dim=2).pow(2).mean() +
                  torch.diff(grid, dim=3).pow(2).mean() +
                  torch.diff(grid, dim=4).pow(2).mean())
            loss = loss + tv_weight * tv
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        losses.append(loss.item())
        xs.append(it)
        if loss.item() < best_loss:
            best_loss = loss.item()
 
        if (it + 1) % log_interval == 0:
            psnr = -10.0 * np.log10(loss.item())
            elapsed = (time.time() - t0) / 60.
            print(f'Iteration {it+1}/{iters}, Loss: {loss.item():.3e}, '
                  f'PSNR: {psnr:.2f} dB, Time: {elapsed:.1f} min')
 
    return {
        'grid': grid.detach(),
        'model_size': num_params,
        'losses': losses,
        'xs': xs,
        'best_loss': best_loss,
    }
 
 
def render_image_from_grid(grid, pose, H, W, focal,
                           near=2., far=6., N_samples=128,
                           bbox_min=None, bbox_max=None,
                           chunk=512, device=device):
    if bbox_min is None:
        bbox_min = np.array([-1.5, -1.5, -1.5], dtype=np.float32)
    if bbox_max is None:
        bbox_max = np.array([ 1.5,  1.5,  1.5], dtype=np.float32)
    bbox_min_t = torch.tensor(bbox_min, dtype=torch.float32, device=device)
    bbox_max_t = torch.tensor(bbox_max, dtype=torch.float32, device=device)
 
    rays_o, rays_d = get_rays(H, W, focal, pose)
    ro_flat = rays_o.reshape(-1, 3)
    rd_flat = rays_d.reshape(-1, 3)
 
    rgb_chunks, depth_chunks = [], []
    with torch.no_grad():
        for ci in range(0, ro_flat.shape[0], chunk):
            ro_c = torch.tensor(ro_flat[ci:ci + chunk],
                                dtype=torch.float32, device=device)
            rd_c = torch.tensor(rd_flat[ci:ci + chunk],
                                dtype=torch.float32, device=device)
            n = ro_c.shape[0]
            z = torch.linspace(near, far, N_samples, device=device) \
                     .unsqueeze(0).expand(n, -1)
            pts = ro_c[:, None, :] + rd_c[:, None, :] * z[:, :, None]
            raw = sample_voxel_grid(grid, pts.reshape(-1, 3),
                                    bbox_min_t, bbox_max_t)
            raw = raw.reshape(n, N_samples, 4)
            rgb_c, depth_c, _ = volume_render(raw, z)
            rgb_chunks.append(rgb_c.cpu().numpy())
            depth_chunks.append(depth_c.cpu().numpy())
 
    rgb_image = np.concatenate(rgb_chunks, 0).reshape(H, W, 3)
    depth_map = np.concatenate(depth_chunks, 0).reshape(H, W)
    return rgb_image, depth_map