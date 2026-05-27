import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets
import pickle
from PIL import Image
import logging
import argparse
import sys
sys.path.append('../../')
from helpers import *
from fns_gap import *
from bandlimited_signal import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='bandlimited', type=str)
  parser.add_argument('--param_to_run', default='feature_dim', type=str)
  parser.add_argument('--operation', default='concatenate', type=str)
  parser.add_argument('--decoder', default='linear', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  args, _ = parser.parse_known_args()
  return args

PARAMS_DICT = {
    'volume_resolution': {
      'volume_resos': [23, 29, 33, 37, 40, 42, 45, 47, 49, 50, 52, 53, 55, 56, 58, 59, 60, 61, 62, 63, 64],
      'lineres': 2,
      'planeres': 2,
      'line_feature_dim': 4,
      'plane_feature_dim': 4,
      'volume_feature_dim': 1,
      'hidden_dim': 64,
    },
    'feature_dim': {
      'feature_dims': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
      'lineres': 64,
      'planeres': 64,
      'volumeres': 2,
      'volume_feature_dim': 1,
      'hidden_dim': 64,
    },
    'RES': 64,
}

# IDX_LIST = [1234, 2026, 5678, 7890, 7618]
IDX_LIST = [1234]

MODEL_NAME = 'ga-planes_eta'

def run_exp(args):
  tol = 1e-6
  learning_rate, iters = 8e-3, 4000
  
  RES = PARAMS_DICT['RES']
  device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'  

  for index in range(len(IDX_LIST)):
    idx = IDX_LIST[index]
    print(f'Start id {idx}')
    if args.dataset=='dragon':
        signal =  resize(Voxel_Fitting(dimension=3, length=RES, bandlimit=0.6, seed=idx, super_resolution=False, sparse=True).signal, (RES, RES, RES))
    elif args.dataset == 'sphere':
        signal = SparseSphereSignal(dimension=3, length=RES, bandlimit=0.3, seed=idx, generate=False).signal
    elif args.dataset == 'bandlimited':
        signal = BandlimitedSignal(dimension=3, length=RES, bandlimit=0.9, seed=idx, generate=False).signal      

    if args.param_to_run == 'volume_resolution':
      # 1. Configure the logger
      logging.basicConfig(
        filename=f"logs/{MODEL_NAME}/{args.dataset}/{args.param_to_run}_i{idx}_s{args.seed}.log",  # Name of the log file
        level=logging.INFO,  # Minimum log level to capture
        format='%(message)s', # Log message format
        filemode='w', # Overwrite the file each time the script runs (use 'a' to append)
        force=True,
      )

      # Create a logger object
      logger = logging.getLogger(__name__)


      gap_args = {
        "lineres": PARAMS_DICT[args.param_to_run]['lineres'],
        "planeres": PARAMS_DICT[args.param_to_run]['planeres'],
        "line_feature_dim": PARAMS_DICT[args.param_to_run]['line_feature_dim'],
        "plane_feature_dim": PARAMS_DICT[args.param_to_run]['plane_feature_dim'],
        "volume_feature_dim": PARAMS_DICT[args.param_to_run]['volume_feature_dim'],
        "hidden_dim": PARAMS_DICT[args.param_to_run]['hidden_dim'],
        "operation": args.operation,
        "decoder": args.decoder,
      }

      volume_resos, exp_dict = PARAMS_DICT[args.param_to_run]['volume_resos'], {}
      for volume_reso in volume_resos:
        logger.info(f"Volume resolution={volume_reso}")
        gap_args['volumeres'] = volume_reso
        max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
        eta, list_of_outputs, i = None, [], 0
        while abs(max_dist - prev_max_dist) > tol:
            output = fit_gaplanes(args=gap_args, img=signal, iters=iters, learning_rate=learning_rate, log_interval = iters+1, seed=args.seed + i * volume_reso, device=device)
            error = np.linalg.norm(signal.flatten() - output['best_pred'].flatten())
            list_of_outputs.append(output['best_pred'].flatten())
            if len(list_of_outputs) >= 2:
                prev_max_dist = max_dist
                max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
            if error < min_error:
                min_error, best_pred = error, output['best_pred'].flatten()
            logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
            i+=1
        logger.info(f"Final eta for volume resolution {volume_reso} is {max_dist}")
        exp_dict[f'{volume_reso}'] = {}
        exp_dict[f'{volume_reso}']['eta'] = eta
        exp_dict[f'{volume_reso}']['eta_upper_bound'] = max_dist
        exp_dict[f'{volume_reso}']['error'] = min_error
        exp_dict[f'{volume_reso}']['pred'] = best_pred
        logger.info("-----------------------------------------------------------")

      with open(f"pkls/{MODEL_NAME}/{args.dataset}/{args.param_to_run}_i{idx}_s{args.seed}.pkl", "wb") as f:
          pickle.dump(exp_dict, f)
    elif args.param_to_run == 'feature_dim':
      # 1. Configure the logger
      logging.basicConfig(
        filename=f"../logs/3d_{args.dataset}/{MODEL_NAME}.log",  # Name of the log file
        level=logging.INFO,  # Minimum log level to capture
        format='%(message)s', # Log message format
        filemode='w', # Overwrite the file each time the script runs (use 'a' to append)
        force=True,
      )

      # Create a logger object
      logger = logging.getLogger(__name__)

      gap_args = {
        "lineres": PARAMS_DICT[args.param_to_run]['lineres'],
        "planeres": PARAMS_DICT[args.param_to_run]['planeres'],
        "volumeres": PARAMS_DICT[args.param_to_run]['volumeres'],
        "volume_feature_dim": PARAMS_DICT[args.param_to_run]['volume_feature_dim'],
        "hidden_dim": PARAMS_DICT[args.param_to_run]['hidden_dim'],
        "operation": args.operation,
        "decoder": args.decoder,
      }

      feature_dims, exp_dict = PARAMS_DICT[args.param_to_run]['feature_dims'], {}
      for feature_dim in feature_dims:
        logger.info(f"Feature Dimension={feature_dim}")
        gap_args['line_feature_dim'] = feature_dim
        gap_args['plane_feature_dim'] = feature_dim
        max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
        eta, list_of_outputs, i = None, [], 0
        while abs(max_dist - prev_max_dist) > tol:
            output = fit_gaplanes(args=gap_args, img=signal, iters=iters, learning_rate=learning_rate, log_interval = iters+1, seed=args.seed + i * feature_dim, device=device)
            error = np.linalg.norm(signal.flatten() - output['best_pred'].flatten())
            list_of_outputs.append(output['best_pred'].flatten())
            if len(list_of_outputs) >= 2:
                prev_max_dist = max_dist
                max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
            if error < min_error:
                min_error, best_pred = error, output['best_pred'].flatten()
            logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
            i+=1
        logger.info(f"Final eta for feature dimension {feature_dim} is {max_dist}")
        exp_dict[f'{feature_dim}'] = {}
        exp_dict[f'{feature_dim}']['eta'] = eta
        exp_dict[f'{feature_dim}']['eta_upper_bound'] = max_dist
        exp_dict[f'{feature_dim}']['error'] = min_error
        exp_dict[f'{feature_dim}']['pred'] = best_pred
        logger.info("-----------------------------------------------------------")

      with open(f"../results/3d_{args.dataset}/{MODEL_NAME}.pkl", "wb") as f:
          pickle.dump(exp_dict, f)
    else:
      print(f"Parameter {args.param_to_run} not recognized. Skipping.")

    print(f'Done with id {idx}')
  # After the loop, you can close the file implicitly, 
  # or use logging.shutdown() to ensure all messages are written.
  logging.shutdown() 

if __name__ == "__main__":
    args = parse_args()
    run_exp(args)
