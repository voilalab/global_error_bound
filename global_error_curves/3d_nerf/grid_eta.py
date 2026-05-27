import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import bidsio
import pickle
import logging
import sys
import argparse
sys.path.append('../../')
from helpers import *
from fns_grid import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  args, _ = parser.parse_known_args()
  return args


def compute_pred_and_error(output, near, far, device, N_samples=512):
    rendered, _ = render_image_from_grid(
        output['grid'], test_poses[TEST_IDX], H, W, focal,
        near=near, far=far, N_samples=N_samples, device=device
    )
    error = np.linalg.norm(test_images[TEST_IDX] - rendered)
    return error, rendered

PARAMS_DICT = {
    'grid_resos': [50, 64, 80, 100, 128, 150, 200, 256],
}

MODEL_NAME = 'grid_eta'

data = np.load('lego_400.npz')
images = data['images']
poses = data['poses']
focal = data['focal']
H, W = images.shape[1:3]

images, val_images, test_images = np.split(images[...,:3], [100,107], axis=0)
poses, val_poses, test_poses = np.split(poses, [100,107], axis=0)


def run_exp(args):
    tol = 1e-6
    learning_rate, iters = 5e-2, 15000
    N_samples = 512
    batch_size = 1024
    near, far = 2., 6.

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'   

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

    grid_resos = PARAMS_DICT['grid_resos']
    exp_dict = {}
    torch.manual_seed(args.seed)

    for grid_reso in grid_resos:
        print(f"Begin grid_reso={grid_reso}")
        logger.info(f"Grid Resolution={grid_reso}")
        max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
        eta, list_of_outputs, i = None, [], 0
        while abs(max_dist - prev_max_dist) > tol:
            output = fit_grid_nerf(
                images, poses, focal, H, W,
                val_images=val_images, val_poses=val_poses,
                r=grid_reso, iters=iters, lr=learning_rate,
                near=near, far=far, N_samples=N_samples, batch_size=batch_size,
                tv_weight=1e-5, log_interval=iters + 1, device=device, seed=args.seed + i * grid_reso
            )
            error, pred = compute_pred_and_error(output, near, far, device, N_samples)
            list_of_outputs.append(pred)
            i+=1
            if i > 5:
                if len(list_of_outputs) >= 2:
                    prev_max_dist = max_dist
                    max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
                if error < min_error:
                    min_error, best_pred = error, pred
                logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
            else:
                logger.info(f"   Iteration {i} with loss={output['best_loss']:.3e} and error={error}")
            
        logger.info(f"Final eta for grid resolution {grid_reso} is {max_dist}")
        exp_dict[f'{grid_reso}'] = {}
        exp_dict[f'{grid_reso}']['eta'] = eta
        exp_dict[f'{grid_reso}']['eta_upper_bound'] = max_dist
        exp_dict[f'{grid_reso}']['error'] = min_error
        exp_dict[f'{grid_reso}']['pred'] = best_pred
        exp_dict[f'{grid_reso}']['model_size'] = output['model_size']
        logger.info("-----------------------------------------------------------")
        print(f"Finish grid_reso={grid_reso}")

    with open(f"../results/3d_nerf/{MODEL_NAME}.pkl", "wb") as f:
        pickle.dump(exp_dict, f)


    # After the loop, you can close the file implicitly, 
    # or use logging.shutdown() to ensure all messages are written.
    logging.shutdown() 

if __name__ == "__main__":
    args = parse_args()
    run_exp(args)
