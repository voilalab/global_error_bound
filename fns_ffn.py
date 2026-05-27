import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from helpers import *

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# Fourier feature mapping
def input_mapping(x, B):
  if B is None: return x
  else:
    x_proj = (2.*torch.pi*x) @ B.T
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FullyConnectedNetwork(nn.Module):
  """A fully connected network with depth=num_layers and width=num_channels """
  def __init__(self, num_layers=4, num_channels=256, input_dim=512, output_dim=1):
    super().__init__()
    layers = [nn.Linear(input_dim, num_channels), nn.ReLU()]
    for i in range(num_layers-1):
      layers.append(nn.Linear(num_channels, num_channels))
      layers.append(nn.ReLU())
    layers.append(nn.Linear(num_channels, output_dim))
    # layers.append(nn.Sigmoid())
    self.network = nn.Sequential(*layers)

  def forward(self, x):
    return self.network(x)


def fit_fourier_features_learn(y_gt, B=None, network_size=(4, 256), iters=2000, learning_rate=1e-4, log_interval=10000, seed=0, device=device, count_params=False, mask=None):
  if B.shape[1] == 1:
    coords = torch.tensor(np.linspace(0.0, 1.0, len(y_gt), endpoint=False), dtype=torch.float32).unsqueeze(-1).to(device)
  elif B.shape[1] == 2:
    coords = torch.tensor(np.stack(np.meshgrid(np.linspace(-1.0, 1.0, y_gt.shape[0]), np.linspace(-1.0, 1.0, y_gt.shape[1])), -1), dtype=torch.float32).reshape(-1, 2).to(device)
  elif B.shape[1] == 3:
    coords = torch.tensor(np.stack(np.meshgrid(np.linspace(-1.0, 1.0, y_gt.shape[0]), np.linspace(-1.0, 1.0, y_gt.shape[1]), np.linspace(-1.0, 1.0, y_gt.shape[2])), -1), dtype=torch.float32).reshape(-1, 3).to(device)

  if mask is not None:
    fft_axes = tuple(range(B.shape[1]))   
    mask_torch = torch.tensor(mask, dtype=torch.complex64, device=device)
    y_train_fft = torch.tensor(np.fft.fftn(y_gt, axes=fft_axes), dtype=torch.complex64, device=device)

  y_train = torch.tensor(y_gt, dtype=torch.float32, device=device).unsqueeze(-1)
  if B is not None:
    input_dim = B.shape[0] * 2
  else:
    input_dim = 1

  torch.manual_seed(seed)
  fcn = FullyConnectedNetwork(
      num_layers=network_size[0],
      num_channels=network_size[1],
      input_dim=input_dim,
  ).to(device)
  if count_params: print(f'Number of parameters: {count_parameters(fcn)}')

  optimizer = optim.Adam(fcn.parameters(), lr=learning_rate)
  losses, xs = [], []
  best_error, best_pred = float('inf'), None
  for i in range(iters):
    optimizer.zero_grad()
    pred = fcn(input_mapping(coords, B))
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
      print(f'Iteration {i+1}/{iters}, Loss: {loss.item():.3e}')

  with torch.no_grad():
    pred = fcn(input_mapping(coords, B)).squeeze().detach().cpu().numpy()
  return {
        'state': fcn.state_dict(),
        'pred': pred,
        'losses': losses,
        'xs': xs,
        'best_pred': best_pred,
        'best_loss': best_error,
    }