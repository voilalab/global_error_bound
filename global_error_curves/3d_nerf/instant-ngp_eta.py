import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets
import bidsio
import pickle
import logging
import time
import sys
import os
import argparse
sys.path.append('../../')
from helpers import *
from fns_ingp import *

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--param_to_run', default='hash_table_size', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  args, _ = parser.parse_known_args()
  return args

def compute_pred_and_error(output, model_config, num_layers, num_channels, near, far, N_samples=512):
    params = hash_parameters(model_config)
    model = Instant_NGP_NeRF(
        num_layers=num_layers,
        num_channels=num_channels,
        params=params,
        seed=0,
        near=near,
        far=far,
        device=device,
    ).to(device)
    model.load_state_dict(output['state'])
    model.eval()
    
    chunk = 512
    with torch.no_grad():
        # Generate rays for this test pose
        ro_v, rd_v = get_rays(H, W, focal, test_poses[TEST_IDX])
        ro_flat = ro_v.reshape(-1, 3)
        rd_flat = rd_v.reshape(-1, 3)

        # Render in chunks
        rgb_chunks = []
        for ci in range(0, ro_flat.shape[0], chunk):
            ro_c = torch.tensor(ro_flat[ci:ci + chunk],
                                dtype=torch.float32, device=device)
            rd_c = torch.tensor(rd_flat[ci:ci + chunk],
                                dtype=torch.float32, device=device)
            n = ro_c.shape[0]
            z = torch.linspace(near, far, N_samples,
                                device=device).unsqueeze(0).expand(n, -1)
            pts_c = ro_c[:, None, :] + rd_c[:, None, :] * z[:, :, None]
            raw = model(pts_c)
            rgb_c, _, _ = volume_render(raw, z)
            rgb_chunks.append(rgb_c.cpu().numpy())

        rendered = np.concatenate(rgb_chunks, 0).reshape(H, W, 3)

    error = np.linalg.norm(test_images[TEST_IDX] - rendered)
    return error, rendered


PARAMS_DICT = {
    'num_levels': {
      'nums_levels': np.arange(1, 10, 1),
      'hash_table_size': 16,
      'base_resolution': 16,
    },
    'hash_table_size': {
      'hash_sizes': [13, 14, 15, 16, 17, 18, 19, 20],
      'num_levels': 13,
      'base_resolution': 16,
    },
}


MODEL_NAME = 'instant-ngp_eta'

data = np.load('lego_400.npz')
images = data['images']
poses = data['poses']
focal = data['focal']
H, W = images.shape[1:3]

images, val_images, test_images = np.split(images[...,:3], [100,107], axis=0)
poses, val_poses, test_poses = np.split(poses, [100,107], axis=0)

def run_exp(args):
  tol = 1e-6
  learning_rate, iters = 1e-2, 50000
  N_samples = 512
  batch_size = 1024
  num_layers, num_channels = 2, 128
  near, far = 2., 6.

  device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'  

  if args.param_to_run == 'num_levels':
    # 1. Configure the logger
    logging.basicConfig(
      filename=f"logs/{MODEL_NAME}/{args.param_to_run}_s{args.seed}.log",  # Name of the log file
      level=logging.INFO,  # Minimum log level to capture
      format='%(message)s', # Log message format
      filemode='w', # Overwrite the file each time the script runs (use 'a' to append)
      force=True,
    )

    # Create a logger object
    logger = logging.getLogger(__name__)

    hash_table_size, base_resolution = PARAMS_DICT[args.param_to_run]['hash_table_size'], PARAMS_DICT[args.param_to_run]['base_resolution']
    nums_levels = PARAMS_DICT[args.param_to_run]['nums_levels']
    exp_dict = {}
    for num_levels in nums_levels:
      print(f"Begin num_levels={num_levels}")
      logger.info(f"Number of levels={num_levels}")
      max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
      eta, list_of_outputs, i = None, [], 0
      while abs(max_dist - prev_max_dist) > tol:
          model_config = (int(num_levels), hash_table_size, base_resolution, 2.0)
          output = fit_instant_ngp_nerf(
              images, poses, focal, H, W,
              val_images, val_poses,
              model_config=model_config,
              iters=iters, learning_rate=learning_rate,
              batch_size=batch_size, N_samples=N_samples,
              near=near, far=far,
              num_layers=num_layers, num_channels=num_channels,
              log_interval=iters + 1, seed=args.seed + i * num_levels, device=device
          )
          error, pred = compute_pred_and_error(output, model_config, num_layers, num_channels, near, far, N_samples)
          list_of_outputs.append(pred)
          if len(list_of_outputs) >= 2:
              prev_max_dist = max_dist
              max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
          if error < min_error:
              min_error, best_pred = error, pred
          logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
          i+=1
      logger.info(f"Final eta for number of levels {num_levels} is {max_dist}")
      exp_dict[f'{num_levels}'] = {}
      exp_dict[f'{num_levels}']['eta'] = eta
      exp_dict[f'{num_levels}']['eta_upper_bound'] = max_dist
      exp_dict[f'{num_levels}']['error'] = min_error
      exp_dict[f'{num_levels}']['pred'] = best_pred
      exp_dict[f'{num_levels}']['model_size'] = output['model_size']
      logger.info("-----------------------------------------------------------")
      print(f"Finish num_levels={num_levels}")

    with open(f"pkls/{MODEL_NAME}/{args.param_to_run}_s{args.seed}.pkl", "wb") as f:
        pickle.dump(exp_dict, f)

  elif args.param_to_run == 'hash_table_size':
      # 1. Configure the logger
      logging.basicConfig(
          filename=f"../logs/3d_nerf/{MODEL_NAME}.log",  # Name of the log file
          level=logging.INFO,  # Minimum log level to capture
          format='%(message)s', # Log message format
          filemode='w', # Overwrite the file each time the script runs (use 'a' to append)
          force=True,
      )

      # Create a logger object
      logger = logging.getLogger(__name__)

      num_levels, base_resolution = PARAMS_DICT[args.param_to_run]['num_levels'], PARAMS_DICT[args.param_to_run]['base_resolution']
      hash_sizes = PARAMS_DICT[args.param_to_run]['hash_sizes']

      exp_dict, tol = {}, 1e-6

      for hash_size in hash_sizes:
        print(f"Begin hash_table_size={hash_size}")
        logger.info(f"Hash Table Size={hash_size}")
        
        max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
        eta, list_of_outputs, i = None, [], 0

        while abs(max_dist - prev_max_dist) > tol:
            model_config = (num_levels, int(hash_size), base_resolution, 2.0)
            output = fit_instant_ngp_nerf(
                images, poses, focal, H, W,
                val_images, val_poses,
                model_config=model_config,
                iters=iters, learning_rate=learning_rate,
                batch_size=batch_size, N_samples=N_samples,
                near=near, far=far,
                num_layers=num_layers, num_channels=num_channels,
                log_interval=iters + 1, seed=args.seed + i * hash_size, device=device
            )
            error, pred = compute_pred_and_error(output, model_config, num_layers, num_channels, near, far, N_samples)
            list_of_outputs.append(pred)
            if len(list_of_outputs) >= 2:
                prev_max_dist = max_dist
                max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
            if error < min_error:
                min_error, best_pred = error, pred
            logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
            i+=1
        logger.info(f"Final eta for hash table size {hash_size} is {max_dist}")
        exp_dict[f'{hash_size}'] = {}
        exp_dict[f'{hash_size}']['eta'] = eta
        exp_dict[f'{hash_size}']['eta_upper_bound'] = max_dist
        exp_dict[f'{hash_size}']['error'] = min_error
        exp_dict[f'{hash_size}']['pred'] = best_pred
        exp_dict[f'{hash_size}']['model_size'] = output['model_size']
        logger.info("-----------------------------------------------------------")
        print(f"Finish hash_table_size={hash_size}")

      with open(f"../results/3d_nerf/{MODEL_NAME}.pkl", "wb") as f:
          pickle.dump(exp_dict, f)
  else:
    print(f"Parameter {args.param_to_run} not recognized. Skipping.")

  # After the loop, you can close the file implicitly, 
  # or use logging.shutdown() to ensure all messages are written.
  logging.shutdown() 

if __name__ == "__main__":
    args = parse_args()
    run_exp(args)