import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import pickle
import logging
import sys
import argparse
sys.path.append('../../')
from helpers import *
from fns_ffn import *
from bandlimited_signal import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='bandlimited', type=str)
  parser.add_argument('--param_to_run', default='width', type=str)
  parser.add_argument('--bandlimit', default=0.6, type=float)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  args, _ = parser.parse_known_args()
  return args

PARAMS_DICT = {
    'mapping_size': {
        'mapping_sizes': [15, 31, 48, 64, 80, 97, 113, 130, 146, 162, 179, 195, 211, 228, 244, 261, 277, 293, 310, 326, 343, 359, 375, 392, 408, 424, 441, 457, 474, 490, 506, 523, 539, 556, 572, 588, 605, 621, 637],
        'num_layers': 1,
        'width': 400,
        'threshold': 0,
    },
    'width': {
        'widths': [4, 8, 12, 17, 21, 25, 30, 34, 38, 43, 47, 51, 56, 60, 64, 68, 73, 77, 81, 86, 90, 94, 99, 103, 107, 112, 116, 120, 124, 129, 133, 137, 142, 146, 150, 155, 159, 163, 168],
        'mapping_size': 1520,
        'num_layers': 1,
        'threshold': 0,
    },
    'depth': {
        'nums_layers': np.arange(1, 8, 1),
        'mapping_size': 256,
        'width': 32,
    },
    'RES': 64,
}

# IDX_LIST = [131, 456, 789, 101, 202]
# IDX_LIST = [1234, 2026, 5678, 7890, 7618]
IDX_LIST = [1234]

MODEL_NAME = 'ffn_eta'

def run_exp(args):
  scale, tol = 20., 1e-6
  learning_rate, iters = 8e-4, 2000
  
  RES = PARAMS_DICT['RES']
  device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'   

  for index in range(len(IDX_LIST)):
    idx = IDX_LIST[index]
    print(f'Start id {idx}')
    if args.dataset=='dragon':
        signal =  resize(Voxel_Fitting(dimension=3, length=RES, bandlimit=args.bandlimit, seed=idx, super_resolution=False, sparse=True).signal, (RES, RES, RES))
    elif args.dataset == 'sphere':
        signal = SparseSphereSignal(dimension=3, length=RES, bandlimit=args.bandlimit, seed=idx, generate=False).signal
    elif args.dataset == 'bandlimited':
        signal = BandlimitedSignal(dimension=3, length=RES, bandlimit=args.bandlimit, seed=idx, generate=False).signal  

    if args.param_to_run == 'mapping_size':
      # 1. Configure the logger
      logging.basicConfig(
        filename=f"logs/{MODEL_NAME}/{args.dataset}/{args.param_to_run}_i{idx}_b{args.bandlimit}_s{args.seed}.log",  # Name of the log file
        level=logging.INFO,  # Minimum log level to capture
        format='%(message)s', # Log message format
        filemode='w', # Overwrite the file each time the script runs (use 'a' to append)
        force=True,
      )

      # Create a logger object
      logger = logging.getLogger(__name__)

      num_layers, width = PARAMS_DICT[args.param_to_run]['num_layers'], PARAMS_DICT[args.param_to_run]['width']
      mapping_sizes = PARAMS_DICT[args.param_to_run]['mapping_sizes']
      exp_dict = {}
      torch.manual_seed(args.seed)
      B = torch.randn(mapping_sizes[-1], len(signal.shape), dtype=torch.float32, device=device) * scale

      for mapping_size in mapping_sizes:
          logger.info(f"mapping_size={mapping_size}")
          max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
          eta, list_of_outputs, i = None, [], 0
          while abs(max_dist - prev_max_dist) > tol:
              output = fit_fourier_features_learn(y_gt=signal, B=B[:mapping_size, :], network_size=(num_layers, width), iters=iters,
                                                                    learning_rate=learning_rate,log_interval=iters+1, seed=args.seed + i * mapping_size, device=device)
              error = np.linalg.norm(signal.flatten() - output['best_pred'].flatten())
              list_of_outputs.append(output['best_pred'].flatten())
              i+=1
              if mapping_size > PARAMS_DICT[args.param_to_run]['threshold'] or i > 5:
                if len(list_of_outputs) >= 2:
                    prev_max_dist = max_dist
                    max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
                if error < min_error:
                    min_error, best_pred = error, output['best_pred'].flatten()
                logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
              else:
                logger.info(f"   Iteration {i} with loss={output['best_loss']:.3e} and error={error}")
              
          logger.info(f"Final eta for mapping size {mapping_size} is {max_dist}")
          exp_dict[f'{mapping_size}'] = {}
          exp_dict[f'{mapping_size}']['eta'] = eta
          exp_dict[f'{mapping_size}']['eta_upper_bound'] = max_dist
          exp_dict[f'{mapping_size}']['error'] = min_error
          exp_dict[f'{mapping_size}']['pred'] = best_pred
          logger.info("-----------------------------------------------------------")

      with open(f"pkls/{MODEL_NAME}/{args.dataset}/{args.param_to_run}_i{idx}_b{args.bandlimit}_s{args.seed}.pkl", "wb") as f:
          pickle.dump(exp_dict, f)

    elif args.param_to_run == 'width':
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

        num_layers, mapping_size = PARAMS_DICT[args.param_to_run]['num_layers'], PARAMS_DICT[args.param_to_run]['mapping_size']
        widths = PARAMS_DICT[args.param_to_run]['widths']

        exp_dict, tol = {}, 1e-6

        torch.manual_seed(args.seed)
        B = torch.randn(mapping_size, len(signal.shape), dtype=torch.float32, device=device) * scale
        for width in widths:

            logger.info(f"width={width}")
            
            max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
            eta, list_of_outputs, i = None, [], 0
            while abs(max_dist - prev_max_dist) > tol:
                output = fit_fourier_features_learn(y_gt=signal, B=B, network_size=(num_layers, width), iters=iters,
                                                                      learning_rate=learning_rate,
                                                                      log_interval=iters+1, seed=args.seed + i * width, device=device)
                error = np.linalg.norm(signal.flatten() - output['best_pred'].flatten())
                list_of_outputs.append(output['best_pred'].flatten())
                i+=1
                if width > PARAMS_DICT[args.param_to_run]['threshold'] or i > 5:
                  if len(list_of_outputs) >= 2:
                      prev_max_dist = max_dist
                      max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
                  if error < min_error:
                      min_error, best_pred = error, output['best_pred'].flatten()
                  logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
                else:
                  logger.info(f"   Iteration {i} with loss={output['best_loss']:.3e} and error={error}")
            logger.info(f"Final eta for width {width} is {max_dist}")
            exp_dict[f'{width}'] = {}
            exp_dict[f'{width}']['eta'] = eta
            exp_dict[f'{width}']['eta_upper_bound'] = max_dist
            exp_dict[f'{width}']['error'] = min_error
            exp_dict[f'{width}']['pred'] = best_pred
            logger.info("-----------------------------------------------------------")

        with open(f"../results/3d_{args.dataset}/{MODEL_NAME}.pkl", "wb") as f:
            pickle.dump(exp_dict, f)

    elif args.param_to_run == 'depth':
        # 1. Configure the logger
        logging.basicConfig(
            filename=f"logs/{MODEL_NAME}/{args.dataset}/{args.param_to_run}_i{idx}_b{args.bandlimit}_s{args.seed}.log",  # Name of the log file
            level=logging.INFO,  # Minimum log level to capture
            format='%(message)s', # Log message format
            filemode='w', # Overwrite the file each time the script runs (use 'a' to append)
            force=True,
        )

        # Create a logger object
        logger = logging.getLogger(__name__)

        width, mapping_size = PARAMS_DICT[args.param_to_run]['width'], PARAMS_DICT[args.param_to_run]['mapping_size']
        nums_layers = PARAMS_DICT[args.param_to_run]['nums_layers']

        exp_dict, tol = {}, 1e-6

        torch.manual_seed(args.seed)
        B = torch.randn(mapping_size, len(signal.shape), dtype=torch.float32, device=device) * scale
        for num_layers in nums_layers:

            logger.info(f"Number of hidden layers={num_layers}")
            
            max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
            eta, list_of_outputs, i = None, [], 0
            while abs(max_dist - prev_max_dist) > tol:
                output = fit_fourier_features_learn(y_gt=signal, B=B, network_size=(num_layers, width), iters=iters,
                                                                      learning_rate=learning_rate,
                                                                      log_interval=iters+1, seed=args.seed + i * num_layers, device=device, mask=mask)
                error = np.linalg.norm(signal.flatten() - output['best_pred'].flatten())
                list_of_outputs.append(output['best_pred'].flatten())
                if len(list_of_outputs) >= 2:
                    prev_max_dist = max_dist
                    max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
                if error < min_error:
                    min_error, best_pred = error, output['best_pred'].flatten()
                logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
                i+=1
            logger.info(f"Final eta for number of hidden layers {num_layers} is {max_dist}")
            exp_dict[f'{num_layers}'] = {}
            exp_dict[f'{num_layers}']['eta'] = eta
            exp_dict[f'{num_layers}']['eta_upper_bound'] = max_dist
            exp_dict[f'{num_layers}']['error'] = min_error
            exp_dict[f'{num_layers}']['pred'] = best_pred
            logger.info("-----------------------------------------------------------")

        with open(f"pkls/{MODEL_NAME}/{args.dataset}/{args.param_to_run}_i{idx}_b{args.bandlimit}_s{args.seed}.pkl", "wb") as f:
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
