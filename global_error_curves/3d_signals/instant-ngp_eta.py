import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import pickle
import logging
import sys
import argparse
sys.path.append('../../')
from helpers import *
from fns_ingp import *
from fns_ffn import *
from bandlimited_signal import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='bandlimited', type=str)
  parser.add_argument('--param_to_run', default='hash_table_size', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  args, _ = parser.parse_known_args()
  return args

PARAMS_DICT = {
    'num_levels': {
      'nums_levels': np.arange(1, 11, 1),
      'hash_table_size': 16,
      'base_resolution': 16,
    },
    'hash_table_size': {
      'hash_sizes': np.arange(1, 21, 1),
      'num_levels': 4,
      'base_resolution': 16,
    },
    'RES': 64,
}

# IDX_LIST = [1234, 2026, 5678, 7890, 7618]
IDX_LIST = [1234]

MODEL_NAME = 'instant-ngp_eta'

def run_exp(args):
  tol = 1e-6
  learning_rate, iters = 8e-4, 4000
  
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

    if args.param_to_run == 'num_levels':
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

      hash_table_size, base_resolution = PARAMS_DICT[args.param_to_run]['hash_table_size'], PARAMS_DICT[args.param_to_run]['base_resolution']
      nums_levels = PARAMS_DICT[args.param_to_run]['nums_levels']
      exp_dict = {}
      for num_levels in nums_levels:
        logger.info(f"Number of levels={num_levels}")
        max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
        eta, list_of_outputs, i = None, [], 0
        while abs(max_dist - prev_max_dist) > tol:
            model_config = (int(num_levels), hash_table_size, base_resolution, 2.0)
            output = fit_instant_ngp(signal, model_config, iters=iters, learning_rate=learning_rate, log_interval = iters+1, seed=args.seed + i * num_levels, device=device)
            error = np.linalg.norm(signal.flatten() - output['best_pred'].flatten())
            list_of_outputs.append(output['best_pred'].flatten())
            if len(list_of_outputs) >= 2:
                prev_max_dist = max_dist
                max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
            if error < min_error:
                min_error, best_pred = error, output['best_pred'].flatten()
            logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
            i+=1
        logger.info(f"Final eta for number of levels {num_levels} is {max_dist}")
        exp_dict[f'{num_levels}'] = {}
        exp_dict[f'{num_levels}']['eta'] = eta
        exp_dict[f'{num_levels}']['eta_upper_bound'] = max_dist
        exp_dict[f'{num_levels}']['error'] = min_error
        exp_dict[f'{num_levels}']['pred'] = best_pred
        logger.info("-----------------------------------------------------------")

      with open(f"pkls/{MODEL_NAME}/{args.dataset}/{args.param_to_run}_i{idx}_s{args.seed}.pkl", "wb") as f:
          pickle.dump(exp_dict, f)

    elif args.param_to_run == 'hash_table_size':
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

        num_levels, base_resolution = PARAMS_DICT[args.param_to_run]['num_levels'], PARAMS_DICT[args.param_to_run]['base_resolution']
        hash_sizes = PARAMS_DICT[args.param_to_run]['hash_sizes']

        exp_dict, tol = {}, 1e-6

        for hash_size in hash_sizes:
          logger.info(f"Hash Table Size={hash_size}")
          
          max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
          eta, list_of_outputs, i = None, [], 0

          while abs(max_dist - prev_max_dist) > tol:
              model_config = (num_levels, int(hash_size), base_resolution, 2.0)
              output = fit_instant_ngp(signal, model_config, iters=iters, learning_rate=learning_rate, log_interval = iters+1, seed=args.seed + i * hash_size, device=device)
              error = np.linalg.norm(signal.flatten() - output['best_pred'].flatten())
              list_of_outputs.append(output['best_pred'].flatten())
              if len(list_of_outputs) >= 2:
                  prev_max_dist = max_dist
                  max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
              if error < min_error:
                  min_error, best_pred = error, output['best_pred'].flatten()
              logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
              i+=1
          logger.info(f"Final eta for hash table size {hash_size} is {max_dist}")
          exp_dict[f'{hash_size}'] = {}
          exp_dict[f'{hash_size}']['eta'] = eta
          exp_dict[f'{hash_size}']['eta_upper_bound'] = max_dist
          exp_dict[f'{hash_size}']['error'] = min_error
          exp_dict[f'{hash_size}']['pred'] = best_pred
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