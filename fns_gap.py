import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
import time
import sys
sys.path.append('../../')
# from triplane_models import *
from helpers import *

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class GAPlanes2D(nn.Module):
    def __init__(self, dim1, dim2, feature_dim1, feature_dim2, m, resolution, operation, decoder, seed=0):
        super(GAPlanes2D, self).__init__()

        self.resx, self.resy = resolution
        self.operation = operation
        self.decoder = decoder

        # Define the feature tensors
        torch.manual_seed(seed)
        self.line_feature_x = nn.Parameter(torch.randn(feature_dim1, dim1)*0.1)
        self.line_feature_y = nn.Parameter(torch.randn(feature_dim1, dim1)*0.1)
        
        if dim2 > 0:
            self.plane_feature = nn.Parameter(torch.randn(feature_dim2, dim2, dim2)*0.01)

        # Define MLP input dimension
        decoder_in_dim = feature_dim1
        if operation == 'add':
            assert feature_dim1 == feature_dim2, "For multiplication, line and plane features must have the same dimension"
            decoder_in_dim = feature_dim1
        elif operation == "concatenate":
            decoder_in_dim = feature_dim1 + feature_dim2 if dim2 > 0 else feature_dim1
        else:
            raise ValueError(f"Invalid operation {self.operation}; expected add or concatenate")

        # Define the decoder
        if decoder == 'linear':
          self.mlp = nn.Sequential(
            nn.Linear(decoder_in_dim, 1),
          )
        elif decoder == 'nonconvex':
          self.mlp = nn.Sequential(
              nn.Linear(decoder_in_dim, m),
              nn.ReLU(inplace=True),
              nn.Linear(m, 1)
          )
        elif decoder == 'convex':
          self.fc1 = nn.Linear(decoder_in_dim, m)
          self.fc2 = nn.Linear(decoder_in_dim, m)
          self.fc2.weight.requires_grad = False

        else:
          raise ValueError(f"Invalid decoder {decoder}; expected linear, nonconvex, or convex")

    def forward(self, coords):
        # Prepare coordinates for grid_sample
        x_coords = coords[..., 0].unsqueeze(-1)  # [batchx, batchy, 1]
        y_coords = coords[..., 1].unsqueeze(-1)  # [batchx, batchy, 1]

        # Scale to [-1, 1] range for grid_sample
        # x_coords = (x_coords * 2 / self.resx) - 1
        # y_coords = (y_coords * 2 / self.resy) - 1
        x_coords = (x_coords / (self.resy - 1)) * 2 - 1
        y_coords = (y_coords / (self.resx - 1)) * 2 - 1

        # Blank coords as filler
        blank = torch.zeros_like(x_coords)  # [batchx, batchy, 1]

        # Combine x and y coordinates
        # gridx = torch.cat((x_coords, blank), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]
        # gridy = torch.cat((y_coords, blank), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]
        gridx = torch.cat((blank, x_coords), dim=-1).unsqueeze(0)  # x drives the H axis
        gridy = torch.cat((blank, y_coords), dim=-1).unsqueeze(0)

        # Interpolate line features using grid_sample
        line_features_x = self.line_feature_x.unsqueeze(0).unsqueeze(-1)  # [1, feature_dim1, dim1, 1]
        line_features_y = self.line_feature_y.unsqueeze(0).unsqueeze(-1)  # [1, feature_dim1, dim1, 1]

        # Get the feature tensors for grid_sample
        feature_x = F.grid_sample(line_features_x, gridx, mode='bilinear', align_corners=True)  # [1, feature_dim1, batchx, batchy]
        feature_y = F.grid_sample(line_features_y, gridy, mode='bilinear', align_corners=True)  # [1, feature_dim1, batchx, batchy]

        # Prepare for 2D interpolation for the plane feature
        sampled_plane_features = 0
        if hasattr(self, 'plane_feature'):
            plane_features = self.plane_feature.unsqueeze(0)  # [1, dim_features, dim2, dim2]
            plane_grid = torch.cat((x_coords, y_coords), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]

            # Sample from the plane feature using grid_sample
            sampled_plane_features = F.grid_sample(plane_features, plane_grid, mode='bilinear', align_corners=True)  # [1, feature_dim2, batchx, batchy]

        # Combine features
        if self.operation == 'concatenate':
            if hasattr(self, 'plane_feature'):
                combined_features = torch.cat([feature_x * feature_y, sampled_plane_features], dim=1) # [1, feature_dim1 + feature_dim2, batchx, batchy]
            else:
                combined_features = feature_x * feature_y  # [1, feature_dim1, batchx, batchy]
        elif self.operation == 'add':
            combined_features = feature_x * feature_y + sampled_plane_features  # [1, dim_features, batchx, batchy]
        else:
            raise ValueError(f"Invalid operation {self.operation}; expected add or concatenate")

        # Reorder axes so this can be fed to the MLP
        combined_features = combined_features.squeeze(0).permute(1, 2, 0)  # [batchx, batchy, decoder_in_dim]

        # Pass through decoder
        if self.decoder == 'linear' or self.decoder == 'nonconvex':
          output = self.mlp(combined_features).squeeze()  # [batchx, batchy]
        else:  # convex
          output = self.fc1(combined_features) * (self.fc2(combined_features) >= 0)  # [batchx, batchy, m]
          output = torch.sum(output, dim=-1)  # [batchx, batchy]

        return output

class GAPlanes3D(nn.Module):
    def __init__(self, dim1, dim2, dim3, feature_dim1, feature_dim2, feature_dim3 , m, resolution, operation, decoder, seed=0):
        super(GAPlanes3D, self).__init__()

        self.resx, self.resy, self.resz = resolution
        self.operation = operation
        self.decoder = decoder

        # Define the feature tensors
        torch.manual_seed(seed)
        self.line_feature_x = nn.Parameter(torch.randn(feature_dim1, dim1)*0.1)
        self.line_feature_y = nn.Parameter(torch.randn(feature_dim1, dim1)*0.1)
        self.line_feature_z = nn.Parameter(torch.randn(feature_dim1, dim1)*0.1)
        
        if dim2 > 0:
            self.plane_feature_xy = nn.Parameter(torch.randn(feature_dim2, dim2, dim2)*0.01)
            self.plane_feature_yz = nn.Parameter(torch.randn(feature_dim2, dim2, dim2)*0.01)
            self.plane_feature_zx = nn.Parameter(torch.randn(feature_dim2, dim2, dim2)*0.01)
        
        if dim3 > 0:
            self.volume_feature = nn.Parameter(torch.randn(feature_dim3, dim3, dim3, dim3)*0.001)

        # Define MLP input dimension
        decoder_in_dim = feature_dim1
        if operation == 'add':
            assert feature_dim1 == feature_dim2, "For addition, line and plane features must have the same dimension"
            assert feature_dim1 == feature_dim3, "For addition, line and volume features must have the same dimension"
            decoder_in_dim = feature_dim1
        elif operation == "concatenate":
            assert feature_dim1 == feature_dim2, "For concatenation, line and plane features must have the same dimension"
            if dim2 > 0 and dim3 > 0:
                decoder_in_dim = feature_dim1 + 3 * feature_dim2 + feature_dim3
            elif dim2 > 0 and dim3 == 0:
                decoder_in_dim = feature_dim1 + 3 * feature_dim2
            elif dim2 == 0 and dim3 > 0:
                decoder_in_dim = feature_dim1 + feature_dim3
            else: 
                decoder_in_dim = feature_dim1
        else:
            raise ValueError(f"Invalid operation {self.operation}; expected add or concatenate")

        # Define the decoder
        if decoder == 'linear':
          self.mlp = nn.Sequential(
            nn.Linear(decoder_in_dim, 1),
          )
        elif decoder == 'nonconvex':
          self.mlp = nn.Sequential(
              nn.Linear(decoder_in_dim, m),
              nn.ReLU(inplace=True),
              nn.Linear(m, 1)
          )
        elif decoder == 'convex':
          self.fc1 = nn.Linear(decoder_in_dim, m)
          self.fc2 = nn.Linear(decoder_in_dim, m)
          self.fc2.weight.requires_grad = False

        else:
          raise ValueError(f"Invalid decoder {decoder}; expected linear, nonconvex, or convex")

    def forward(self, coords):
        orig_shape = coords.shape[:-1]          # (xdim, ydim, zdim)
        coords = coords.reshape(-1, 3)          # [N, 3]
        N = coords.shape[0]

        # Prepare coordinates for grid_sample
        x_coords = coords[..., 0].unsqueeze(-1)  # [batchx, batchy, 1]
        y_coords = coords[..., 1].unsqueeze(-1)  # [batchx, batchy, 1]
        z_coords = coords[..., 2].unsqueeze(-1)  # [batchx, batchy, 1]

        # Scale to [-1, 1] range for grid_sample
        x_coords = (x_coords / (self.resx - 1)) * 2 - 1
        y_coords = (y_coords / (self.resy - 1)) * 2 - 1
        z_coords = (z_coords / (self.resz - 1)) * 2 - 1

        # Blank coords as filler
        blank = torch.zeros_like(x_coords)  # [batchx, batchy, 1]

        # Combine x and y coordinates
        # gridx = torch.cat((blank, x_coords), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]
        # gridy = torch.cat((blank, y_coords), dim=-1).unsqueeze(0) # [1, batchx, batchy, 2]
        # gridz = torch.cat((blank, z_coords), dim=-1).unsqueeze(0) # [1, batchx, batchy, 2]
        gridx = torch.cat((blank, x_coords), dim=-1).view(1, N, 1, 2)
        gridy = torch.cat((blank, y_coords), dim=-1).view(1, N, 1, 2)
        gridz = torch.cat((blank, z_coords), dim=-1).view(1, N, 1, 2)

        # Interpolate line features using grid_sample
        line_features_x = self.line_feature_x.unsqueeze(0).unsqueeze(-1)  # [1, feature_dim1, dim1, 1]
        line_features_y = self.line_feature_y.unsqueeze(0).unsqueeze(-1)  # [1, feature_dim1, dim1, 1]
        line_features_z = self.line_feature_z.unsqueeze(0).unsqueeze(-1)  # [1, feature_dim1, dim1, 1]

        # Get the feature tensors for grid_sample
        feature_x = F.grid_sample(line_features_x, gridx, mode='bilinear', align_corners=True).squeeze(0).squeeze(-1).transpose(0, 1)  # [1, feature_dim1, batchx, batchy]
        feature_y = F.grid_sample(line_features_y, gridy, mode='bilinear', align_corners=True).squeeze(0).squeeze(-1).transpose(0, 1)  # [1, feature_dim1, batchx, batchy]
        feature_z = F.grid_sample(line_features_z, gridz, mode='bilinear', align_corners=True).squeeze(0).squeeze(-1).transpose(0, 1)  # [1, feature_dim1, batchx, batchy]


        # Prepare for 2D interpolation for the plane feature
        sampled_plane_features_xy, sampled_plane_features_yz, sampled_plane_features_zx = 0, 0, 0
        if hasattr(self, 'plane_feature_xy'):
            plane_features_xy = self.plane_feature_xy.unsqueeze(0)  # [1, feature_dim2, dim2, dim2]
            plane_features_yz = self.plane_feature_yz.unsqueeze(0)  # [1, feature_dim2, dim2, dim2]
            plane_features_zx = self.plane_feature_zx.unsqueeze(0)  # [1, feature_dim2, dim2, dim2]

            # plane_grid_xy = torch.cat((x_coords, y_coords), dim=-1).unsqueeze(0)  # [1, batchx, batchy, 2]
            # plane_grid_yz = torch.cat((y_coords, z_coords), dim=-1).unsqueeze(0)  # [1, batchy, batchz, 2]
            # plane_grid_zx = torch.cat((z_coords, x_coords), dim=-1).unsqueeze(0)  # [1, batchz, batchx, 2]
            grid_xy = torch.cat((x_coords, y_coords), dim=-1).view(1, N, 1, 2)
            grid_yz = torch.cat((y_coords, z_coords), dim=-1).view(1, N, 1, 2)
            grid_zx = torch.cat((z_coords, x_coords), dim=-1).view(1, N, 1, 2)

            # Sample from the plane feature using grid_sample
            sampled_plane_features_xy = F.grid_sample(plane_features_xy, grid_xy, mode='bilinear', align_corners=True).squeeze(0).squeeze(-1).transpose(0, 1)  # [1, feature_dim2, batchx, batchy]
            sampled_plane_features_yz = F.grid_sample(plane_features_yz, grid_yz, mode='bilinear', align_corners=True).squeeze(0).squeeze(-1).transpose(0, 1)  # [1, feature_dim2, batchy, batchz]
            sampled_plane_features_zx = F.grid_sample(plane_features_zx, grid_zx, mode='bilinear', align_corners=True).squeeze(0).squeeze(-1).transpose(0, 1)  # [1, feature_dim2, batchz, batchx]
        
        if hasattr(self, 'volume_feature'):
            volume_features = self.volume_feature.unsqueeze(0)  # [1, feature_dim3, dim3, dim3, dim3]
            # volume_grid = torch.cat((x_coords, y_coords, z_coords), dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, batchx, batchy, 3]
            volume_grid = torch.cat((x_coords, y_coords, z_coords), dim=-1).view(1, N, 1, 1, 3)  # <-- FIX
            sampled_volume_features = F.grid_sample(volume_features, volume_grid, mode='bilinear', align_corners=True).squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1)  # [1, feature_dim3, batchx, batchy, batchz]

        # Combine features
        if self.operation == 'concatenate':
            combined_lines = feature_x * feature_y * feature_z
            combined_planes_xy, combined_planes_yz, combined_planes_zx = sampled_plane_features_xy * feature_z, sampled_plane_features_yz * feature_x, sampled_plane_features_zx * feature_y
            combined_features = torch.cat([combined_lines, combined_planes_xy, combined_planes_yz, combined_planes_zx, sampled_volume_features], dim=1) # [1, feature_dim1 + 3 * feature_dim2 + feature_dim3, batchx, batchy, batchz]
        elif self.operation == 'add':
            lines = feature_x * feature_y * feature_z
            planes = sampled_plane_features_xy * feature_z + sampled_plane_features_yz * feature_x + sampled_plane_features_zx * feature_y
            combined_features = lines + planes + sampled_volume_features  # [1, feature_dim1, batchx, batchy, batchz]
        else:
            raise ValueError(f"Invalid operation {self.operation}; expected add or concatenate")

        # Reorder axes so this can be fed to the MLP
        # combined_features = combined_features.squeeze(0).permute(1, 2, 3, 0)  # [batchx, batchy, batchz, dim_features]

        # Pass through decoder
        if self.decoder == 'linear' or self.decoder == 'nonconvex':
          output = self.mlp(combined_features).squeeze(-1)  # [batchx, batchy, batchz]
        else:  # convex
          output = self.fc1(combined_features) * (self.fc2(combined_features) >= 0)  # [batchx, batchy, batchz, m]
          output = torch.sum(output, dim=-1)  # [batchx, batchy, batchz]

        return output.view(*orig_shape)
    
def fit_gaplanes(args, img, iters=2000, learning_rate=1e-4, log_interval=10000, seed=0, device=device, mask=None, count_params=False):
    if img.ndim == 2:
        xdim, ydim = img.shape

        y_indices, x_indices = np.indices((xdim, ydim))
        coords = np.stack((x_indices, y_indices), axis=-1)  # [xdim, ydim, 2]

        coords = torch.from_numpy(coords).float().to(device)
        targets = torch.from_numpy(img).float().to(device)

        model = GAPlanes2D(dim1=args['lineres'], dim2=args['planeres'], feature_dim1=args['line_feature_dim'], feature_dim2=args['plane_feature_dim'], m=args['hidden_dim'], resolution=img.shape, operation=args['operation'], decoder=args['decoder'], seed=seed).to(device)
    elif img.ndim == 3:
        xdim, ydim, zdim = img.shape

        z_indices, y_indices, x_indices = np.indices((xdim, ydim, zdim))
        coords = np.stack((x_indices, y_indices, z_indices), axis=-1)  # [xdim, ydim, zdim, 3]

        coords = torch.from_numpy(coords).float().to(device)
        targets = torch.from_numpy(img).float().to(device)

        model = GAPlanes3D(dim1=args['lineres'], dim2=args['planeres'], dim3=args['volumeres'], feature_dim1=args['line_feature_dim'], feature_dim2=args['plane_feature_dim'], feature_dim3=args['volume_feature_dim'], m=args['hidden_dim'], resolution=img.shape, operation=args['operation'], decoder=args['decoder'], seed=seed).to(device)
    else:
        raise ValueError(f"Unsupported image dimension {img.ndim}; expected 2 or 3")
    if count_params: print(f'Number of parameters: {count_parameters(model)}')
    if mask is not None:
        fft_axes = tuple(range(len(img.shape)))   
        mask_torch = torch.tensor(mask, dtype=torch.complex64, device=device)
        y_train_fft = torch.tensor(np.fft.fftn(img, axes=fft_axes), dtype=torch.complex64, device=device)

    # print(f'Number of parameters: {count_parameters(model)}')
    # Define loss function and optimizer
    # criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses, xs = [], []
    best_error, best_pred = float('inf'), None
    # Training loop (full batch)
    for iter in range(iters):
        model.train()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(coords)

        # Compute loss
        # loss = criterion(outputs, targets)
        if mask is not None:
            fft_output = torch.fft.fftn(outputs.reshape(*targets.shape), dim=fft_axes)
            masked = fft_output * mask_torch
            loss = 0.5 * torch.mean(torch.abs(masked - y_train_fft) ** 2)
        else:
            loss = torch.mean((outputs.flatten() - targets.flatten()) ** 2)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        xs.append(iter)
        if loss.item() < best_error:
            best_error = loss.item()
            best_pred = outputs.detach().cpu().numpy()

        if (iter+1) % log_interval == 0:
            print(f'Iteration {iter+1}/{iters}, Loss: {loss.item():.3e}, PSNR: {-10*np.log10(loss.item()):.4f}')
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_outputs = model(coords)

    return {
        'pred': final_outputs,
        'losses': losses,
        'xs': xs,
        'best_pred': best_pred,
        'best_loss': best_error,
    }


class GAPlanes3DNeRF(nn.Module):
    def __init__(self, dim1, dim2, dim3,
                 feature_dim1, feature_dim2, feature_dim3,
                 m, operation,
                 aabb_min=(-1., -1., -1.), aabb_max=(1., 1., 1.),
                 num_decoder_layers=2, output_dim=4, seed=0,
                 multires_factors=None):
        super().__init__()
 
        self.operation = operation
        self.output_dim = output_dim
 
        # Multiresolution factors — default [1] gives identical behaviour to
        # the single-resolution version.
        if multires_factors is None:
            multires_factors = [1]
        self.multires_factors = list(multires_factors)
        self.n_scales = len(self.multires_factors)
 
        self.register_buffer('aabb_min', torch.tensor(aabb_min, dtype=torch.float32))
        self.register_buffer('aabb_max', torch.tensor(aabb_max, dtype=torch.float32))
 
        # ── Feature grids ────────────────────────────────────────────────────
        # One set of line + plane grids per resolution scale.
        torch.manual_seed(seed)
 
        self.line_features_x = nn.ParameterList()
        self.line_features_y = nn.ParameterList()
        self.line_features_z = nn.ParameterList()
 
        self.has_planes = dim2 > 0
        if self.has_planes:
            self.plane_features_xy = nn.ParameterList()
            self.plane_features_yz = nn.ParameterList()
            self.plane_features_zx = nn.ParameterList()
 
        for mk in self.multires_factors:
            line_res = mk * dim1
            self.line_features_x.append(nn.Parameter(torch.randn(feature_dim1, line_res) * 0.1))
            self.line_features_y.append(nn.Parameter(torch.randn(feature_dim1, line_res) * 0.1))
            self.line_features_z.append(nn.Parameter(torch.randn(feature_dim1, line_res) * 0.1))
 
            if self.has_planes:
                plane_res = mk * dim2
                self.plane_features_xy.append(nn.Parameter(torch.randn(feature_dim2, plane_res, plane_res) * 0.01))
                self.plane_features_yz.append(nn.Parameter(torch.randn(feature_dim2, plane_res, plane_res) * 0.01))
                self.plane_features_zx.append(nn.Parameter(torch.randn(feature_dim2, plane_res, plane_res) * 0.01))
 
        self.has_volume = dim3 > 0
        if self.has_volume:
            self.volume_feature = nn.Parameter(torch.randn(feature_dim3, dim3, dim3, dim3) * 0.001)
 
        if operation == 'add':
            per_scale_dim = feature_dim1
        elif operation == 'concatenate':
            per_scale_dim = feature_dim1
            if self.has_planes:
                per_scale_dim += 3 * feature_dim2
        else:
            raise ValueError(f"Invalid operation {operation}")
 
        decoder_in_dim = self.n_scales * per_scale_dim
        if self.has_volume:
            decoder_in_dim += feature_dim3
 
        layers = []
        in_d = decoder_in_dim
        for _ in range(num_decoder_layers - 1):
            layers += [nn.Linear(in_d, m), nn.ReLU(inplace=True)]
            in_d = m
        layers.append(nn.Linear(in_d, output_dim))
        self.mlp = nn.Sequential(*layers)
 
    def _normalise(self, coords):
        """Map scene-space coords to [-1, 1] using AABB bounds."""
        return (coords - self.aabb_min) / (self.aabb_max - self.aabb_min) * 2 - 1
 
    @staticmethod
    def _sample_line(param, grid_1d):
        """param: [C, R],  grid_1d: [1, N, 1, 2] → [N, C]"""
        feat = param.unsqueeze(0).unsqueeze(-1)          # [1, C, R, 1]
        return F.grid_sample(feat, grid_1d, mode='bilinear',
                             align_corners=True).squeeze(0).squeeze(-1).transpose(0, 1)
 
    @staticmethod
    def _sample_plane(param, grid_2d):
        """param: [C, R, R],  grid_2d: [1, N, 1, 2] → [N, C]"""
        feat = param.unsqueeze(0)                         # [1, C, R, R]
        return F.grid_sample(feat, grid_2d, mode='bilinear',
                             align_corners=True).squeeze(0).squeeze(-1).transpose(0, 1)
 
    def forward(self, coords):
        orig_shape = coords.shape[:-1]
        coords = coords.reshape(-1, 3)
        N = coords.shape[0]
 
        normed = self._normalise(coords)
        x_coords = normed[:, 0:1]
        y_coords = normed[:, 1:2]
        z_coords = normed[:, 2:3]
 
        blank = torch.zeros_like(x_coords)
 
        gridx = torch.cat((blank, x_coords), dim=-1).view(1, N, 1, 2)
        gridy = torch.cat((blank, y_coords), dim=-1).view(1, N, 1, 2)
        gridz = torch.cat((blank, z_coords), dim=-1).view(1, N, 1, 2)
 
        if self.has_planes:
            grid_xy = torch.cat((x_coords, y_coords), dim=-1).view(1, N, 1, 2)
            grid_yz = torch.cat((y_coords, z_coords), dim=-1).view(1, N, 1, 2)
            grid_zx = torch.cat((z_coords, x_coords), dim=-1).view(1, N, 1, 2)
 
        scale_features = []
        for s in range(self.n_scales):
            fx = self._sample_line(self.line_features_x[s], gridx)   # [N, d1]
            fy = self._sample_line(self.line_features_y[s], gridy)
            fz = self._sample_line(self.line_features_z[s], gridz)
 
            if self.has_planes:
                pxy = self._sample_plane(self.plane_features_xy[s], grid_xy)  # [N, d2]
                pyz = self._sample_plane(self.plane_features_yz[s], grid_yz)
                pzx = self._sample_plane(self.plane_features_zx[s], grid_zx)
 
            if self.operation == 'concatenate':
                parts = [fx * fy * fz]
                if self.has_planes:
                    parts += [pxy * fz, pyz * fx, pzx * fy]
                scale_features.append(torch.cat(parts, dim=1))
            elif self.operation == 'add':
                combined = fx * fy * fz
                if self.has_planes:
                    combined = combined + pxy * fz + pyz * fx + pzx * fy
                scale_features.append(combined)
 
        # Concatenate across resolution scales
        combined = torch.cat(scale_features, dim=1)          # [N, S * per_scale_dim]
 
        # Volume features (single resolution, appended once)
        if self.has_volume:
            vol_grid = torch.cat((x_coords, y_coords, z_coords), dim=-1).view(1, N, 1, 1, 3)
            sv = F.grid_sample(self.volume_feature.unsqueeze(0), vol_grid,
                               mode='bilinear', align_corners=True
                               ).squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1)
            combined = torch.cat([combined, sv], dim=1)       # [N, S * per_scale + d3]
 
        output = self.mlp(combined)  # [N, output_dim]
        return output.view(*orig_shape, self.output_dim)
    
def fit_gaplanes_nerf(args, images, poses, focal, H, W,
                      val_images=None, val_poses=None,
                      iters=50000, learning_rate=5e-4, batch_size=1024,
                      N_samples=512, near=2., far=6., stratified=True,
                      log_interval=5000, seed=0, device=device, count_params=False):
 
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
 
    aabb_min = (-far, -far, -far)
    aabb_max = ( far,  far,  far)
 
    torch.manual_seed(seed)
    model = GAPlanes3DNeRF(
        dim1=args['lineres'],
        dim2=args.get('planeres', 0),
        dim3=args.get('volumeres', 0),
        feature_dim1=args['line_feature_dim'],
        feature_dim2=args.get('plane_feature_dim', 0),
        feature_dim3=args.get('volume_feature_dim', 0),
        m=args['hidden_dim'],
        operation=args['operation'],
        aabb_min=aabb_min, aabb_max=aabb_max,
        num_decoder_layers=args.get('num_decoder_layers', 2),
        output_dim=4,
        seed=seed,
        multires_factors=args.get('multires_factors', None),
    ).to(device)
 
    n_params = sum(p.numel() for p in model.parameters())
    if count_params:
        print(f'GA-Planes NeRF — {n_params:,} parameters')
 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses, xs = [], []
    best_loss = float('inf')
    b_i = 0
    t0 = time.time()
 
    for i in range(iters):
        if b_i + batch_size > all_rays_o.shape[0]:
            b_i = 0
        ro = torch.tensor(all_rays_o[b_i:b_i+batch_size], dtype=torch.float32, device=device)
        rd = torch.tensor(all_rays_d[b_i:b_i+batch_size], dtype=torch.float32, device=device)
        target = torch.tensor(all_rgb[b_i:b_i+batch_size], dtype=torch.float32, device=device)
        b_i += batch_size
 
        z_vals = torch.linspace(near, far, N_samples, device=device)
        if stratified:
            z_vals = z_vals + torch.rand(batch_size, N_samples, device=device) * (far - near) / N_samples
        else:
            z_vals = z_vals.unsqueeze(0).expand(batch_size, -1)
        pts = ro[:, None, :] + rd[:, None, :] * z_vals[:, :, None]  # [B, N_samples, 3]
 
        optimizer.zero_grad()
        raw = model(pts.reshape(-1, 3))        # [B*N_samples, 4]
        raw = raw.reshape(batch_size, N_samples, 4)
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
            print(f'Iter {i+1}/{iters}  loss={loss.item():.3e}  PSNR={psnr:.2f} dB  '
                  f'time={((time.time()-t0)/60.):.1f} min')
 
    return {
        'state': model.state_dict(),
        'model_size': n_params,
        'losses': losses,
        'xs': xs,
        'best_loss': best_loss,
    }