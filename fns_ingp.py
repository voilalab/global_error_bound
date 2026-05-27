import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import tinycudann as tcnn
import time
from helpers import *

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def hash_parameters(config):
  return {
        "encoding": {
            "otype": "Grid",
            "type": "Hash",
            "n_levels":          config[0],
            "n_features_per_level": 2,
            "log2_hashmap_size": config[1],
            "base_resolution":   config[2],
            "per_level_scale":   config[3],
        },
        "network": {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": 64,
            "n_hidden_layers": 2,
        },
  }

def make_network_with_input_encoding(params, seed=0):
  torch.manual_seed(seed)
  np.random.seed(seed)
  return tcnn.NetworkWithInputEncoding(
      n_input_dims = 2,
      n_output_dims = 1,
      encoding_config = params["encoding"],
      network_config = params["network"]
  )

class Instant_NGP(nn.Module):
  """A fully connected network with depth=num_layers and width=num_channels """
  def __init__(self, num_layers=2, num_channels=64, input_dim=2, output_dim=1, params=None, seed=0, device=device):
    super().__init__()
    torch.manual_seed(seed)
    self.encoding = tcnn.Encoding(input_dim, params["encoding"], dtype=torch.float32).to(device)
    layers = [nn.Linear(self.encoding.n_output_dims, num_channels), nn.ReLU()]
    for i in range(num_layers-1):
      layers.append(nn.Linear(num_channels, num_channels))
      layers.append(nn.ReLU())
    layers.append(nn.Linear(num_channels, output_dim))
    self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(self.encoding(x))


def fit_instant_ngp(y_gt, model_config, iters=2000, learning_rate=1e-4, log_interval=500, seed=0, device=device, mask=None, count_params=False):
  torch.manual_seed(seed)
  np.random.seed(seed)
  if len(y_gt.shape) == 1:
    coords = torch.tensor(np.linspace(-1.0, 1.0, len(y_gt), endpoint=False), dtype=torch.float32).unsqueeze(-1).to(device)
    coords = torch.cat([coords, torch.zeros_like(coords)], dim=-1)
  elif len(y_gt.shape) == 2:
    coords = torch.tensor(np.stack(np.meshgrid(np.linspace(-1.0, 1.0, y_gt.shape[0]), np.linspace(-1.0, 1.0, y_gt.shape[1])), -1), dtype=torch.float32).reshape(-1, 2).to(device)
  elif len(y_gt.shape) == 3:
    coords = torch.tensor(np.stack(np.meshgrid(np.linspace(-1.0, 1.0, y_gt.shape[0]), np.linspace(-1.0, 1.0, y_gt.shape[1]), np.linspace(-1.0, 1.0, y_gt.shape[2])), -1), dtype=torch.float32).reshape(-1, 3).to(device)

  if mask is not None:
    fft_axes = tuple(range(len(y_gt.shape)))   
    mask_torch = torch.tensor(mask, dtype=torch.complex64, device=device)
    y_train_fft = torch.tensor(np.fft.fftn(y_gt, axes=fft_axes), dtype=torch.complex64, device=device)

  y_train = torch.tensor(y_gt, dtype=torch.float32, device=device).unsqueeze(-1)
  model = Instant_NGP(num_layers=2, num_channels=64, params=hash_parameters(model_config), input_dim=len(y_gt.shape), seed=seed, device=device).to(device)
  if count_params: print(f'Number of parameters: {count_parameters(model)}')
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  losses, xs = [], []
  best_error, best_pred = float('inf'), None
  for i in range(iters):
    optimizer.zero_grad()
    pred = model(coords)
    if mask is not None:
      fft_output = torch.fft.fftn(pred.reshape(*y_gt.shape), dim=fft_axes)
      masked = fft_output * mask_torch
      loss = 0.5 * torch.mean(torch.abs(masked - y_train_fft) ** 2)
    else:
      loss = torch.mean((pred.flatten() - y_train.flatten()) ** 2)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    xs.append(i)
    if loss.item() < best_error:
      best_error = loss.item()
      best_pred = pred.squeeze().detach().cpu().numpy()
    if (i+1) % log_interval == 0:
      print(f'Iteration {i+1}/{iters}, Loss: {loss.item():.3e}, PSNR: {psnr_from_mse(pred, y_train):.2f} dB')

  with torch.no_grad():
    pred = model(coords).squeeze().cpu().numpy()
  return {
        'state': model.state_dict(),
        'pred': pred,
        'losses': losses,
        'xs': xs,
        'best_pred': best_pred,
        'best_loss': best_error,
    }


class Instant_NGP_NeRF(nn.Module):
    def __init__(self, num_layers=2, num_channels=64, params=None,
                 seed=0, near=2.0, far=6.0, device=device):
        super().__init__()
        torch.manual_seed(seed)
        self.near = near
        self.far = far
        self.encoding = tcnn.Encoding(
            3, params["encoding"], dtype=torch.float32
        ).to(device)
        layers = [nn.Linear(self.encoding.n_output_dims, num_channels), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_channels, num_channels))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(num_channels, 4))  # 3 rgb + 1 sigma
        self.network = nn.Sequential(*layers)
 
    def forward(self, pts):
        shape = pts.shape[:-1]
        flat = pts.reshape(-1, 3)
        normed = normalize_pts(flat, self.near, self.far)
        normed = normed.clamp(0.0, 1.0)  # safety clamp for tcnn
        out = self.network(self.encoding(normed))
        return out.reshape(*shape, 4)
    
def fit_instant_ngp_nerf(images, poses, focal, H, W,
                         val_images, val_poses,
                         model_config,
                         iters=50000, learning_rate=1e-2,
                         batch_size=1024, N_samples=128,
                         near=2.0, far=6.0, stratified=True,
                         num_layers=2, num_channels=64,
                         log_interval=5000, seed=0,
                         device=device, count_params=False):

    all_rays_o, all_rays_d, all_rgb = [], [], []
    for i in range(poses.shape[0]):
        ro, rd = get_rays(H, W, focal, poses[i])
        all_rays_o.append(ro.reshape(-1, 3))
        all_rays_d.append(rd.reshape(-1, 3))
        all_rgb.append(images[i].reshape(-1, 3))
    all_rays_o = np.concatenate(all_rays_o, 0)
    all_rays_d = np.concatenate(all_rays_d, 0)
    all_rgb    = np.concatenate(all_rgb, 0)
 
    rng = np.random.RandomState(seed)
    perm = rng.permutation(all_rays_o.shape[0])
    all_rays_o, all_rays_d, all_rgb = all_rays_o[perm], all_rays_d[perm], all_rgb[perm]
    

    torch.manual_seed(seed)
    np.random.seed(seed)
 
    params = hash_parameters(model_config)
    model = Instant_NGP_NeRF(
        num_layers=num_layers,
        num_channels=num_channels,
        params=params,
        seed=seed,
        near=near,
        far=far,
        device=device,
    ).to(device)
 
    n_params = sum(p.numel() for p in model.parameters())
    if count_params:
        print(f'Number of parameters: {n_params}')
 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-15)
    losses, xs = [], []
    best_loss = float('inf')
    b_i = 0
    t0 = time.time()
 
    for i in range(iters):
        if b_i + batch_size > all_rays_o.shape[0]:
            b_i = 0
        ro = torch.tensor(all_rays_o[b_i:b_i + batch_size], dtype=torch.float32, device=device)
        rd = torch.tensor(all_rays_d[b_i:b_i + batch_size], dtype=torch.float32, device=device)
        target = torch.tensor(all_rgb[b_i:b_i + batch_size], dtype=torch.float32, device=device)
        b_i += batch_size
 
        z_vals = torch.linspace(near, far, N_samples, device=device)
        if stratified:
            z_vals = z_vals + torch.rand(batch_size, N_samples, device=device) * (far - near) / N_samples
        else:
            z_vals = z_vals.unsqueeze(0).expand(batch_size, -1)
        pts = ro[:, None, :] + rd[:, None, :] * z_vals[:, :, None]  # [B, N_samples, 3]
 
        optimizer.zero_grad()
        raw = model(pts)  # [B, N_samples, 4]
        rgb_map, _, _ = volume_render(raw, z_vals)
 
        loss = torch.mean((rgb_map - target) ** 2)
        loss.backward()
        optimizer.step()
 
        losses.append(loss.item())
        xs.append(i)
        if loss.item() < best_loss:
            best_loss = loss.item()
 
        if (i + 1) % log_interval == 0:
            psnr = -10.0 * np.log10(loss.item())
            elapsed = (time.time() - t0) / 60.0
            print(f'Iteration {i+1}/{iters}, Loss: {loss.item():.3e}, '
                  f'PSNR: {psnr:.2f} dB, Time: {elapsed:.1f} min')
 
    return {
        'state': model.state_dict(),
        'model_size': n_params,
        'losses': losses,
        'xs': xs,
        'best_loss': best_loss,
    }