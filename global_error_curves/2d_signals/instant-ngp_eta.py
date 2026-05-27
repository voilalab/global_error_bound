import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets
from PIL import Image
import pickle
import bidsio
import logging
import sys
import argparse
sys.path.append('../../')
from helpers import *
from fns_ingp import *
from bandlimited_signal import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='div2k', type=str)
  parser.add_argument('--param_to_run', default='hash_table_size', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  args, _ = parser.parse_known_args()
  return args

PARAMS_DICT = {
  'div2k': {
    'num_levels': {
      'nums_levels': np.arange(1, 9, 1),
      'hash_table_size': 11,
      'base_resolution': 16,
    },
    'hash_table_size': {
      'hash_sizes': np.arange(1, 15, 1),
      'num_levels': 5,
      'base_resolution': 16,
    },
    'RES': 128,
  },
  'celeba': {
    'num_levels': {
      'nums_levels': np.arange(1, 9, 1),
      'hash_table_size': 11,
      'base_resolution': 16,
    },
    'hash_table_size': {
      'hash_sizes': np.arange(1, 15, 1),
      'num_levels': 5,
      'base_resolution': 16,
    },
    'RES': 128,
  },
  'cifar10': {
    'num_levels': {
      'nums_levels': np.arange(1, 8, 1),
      'hash_table_size': 7,
      'base_resolution': 16,
    },
    'hash_table_size': {
      'hash_sizes': np.arange(1, 11, 1),
      'num_levels': 3,
      'base_resolution': 16,
    },
    'RES': 32,
  },
  'cifar100': {
    'num_levels': {
      'nums_levels': np.arange(1, 8, 1),
      'hash_table_size': 7,
      'base_resolution': 16,
    },
    'hash_table_size': {
      'hash_sizes': np.arange(1, 11, 1),
      'num_levels': 3,
      'base_resolution': 16,
    },
    'RES': 32,
  },
  'mnist': {
    'num_levels': {
      'nums_levels': np.arange(1, 8, 1),
      'hash_table_size': 7,
      'base_resolution': 16,
    },
    'hash_table_size': {
      'hash_sizes': np.arange(1, 11, 1),
      'num_levels': 3,
      'base_resolution': 16,
    },
    'RES': 32,
  },
  'sphere': {
    'num_levels': {
      'nums_levels': np.arange(1, 9, 1),
      'hash_table_size': 12,
      'base_resolution': 16,
    },
    'hash_table_size': {
      'hash_sizes': np.arange(1, 16, 1),
      'num_levels': 5,
      'base_resolution': 16,
    },
    'bandlimit': 0.3,
  },
  'bandlimited': {
    'num_levels': {
      'nums_levels': np.arange(1, 9, 1),
      'hash_table_size': 12,
      'base_resolution': 16,
    },
    'hash_table_size': {
      'hash_sizes': np.arange(1, 16, 1),
      'num_levels': 5,
      'base_resolution': 16,
    },
    'bandlimit': 0.9,
  },
  'mri': {
    'num_levels': {
      'nums_levels': np.arange(1, 9, 1),
      'hash_table_size': 11,
      'base_resolution': 16,
    },
    'hash_table_size': {
      'hash_sizes': np.arange(1, 15, 1),
      'num_levels': 5,
      'base_resolution': 16,
    },
    'RES': 128,
  },
}

IDX_LIST = [131, 456, 789, 101, 202]
# SEED_LIST = [1234, 2026, 5678, 7890, 7618]
MRI_IDX_LIST =  [131, 56, 89, 101, 202]
SEED_LIST = [1234]

MODEL_NAME = 'instant-ngp_eta'

def run_exp(args):
  tol = 1e-6
  learning_rate, iters = 5e-4, 10000
  
  dataset_dict = PARAMS_DICT[args.dataset]
  device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'  
  mask = None
  if args.dataset == 'mri':
    bids_loader = bidsio.BIDSLoader(data_entities=[{'subject': '',
                                              'session': '',
                                              'suffix': 'T1w',
                                              'space': 'MNI152NLin2009aSym'}],
                              target_entities=[],
                              data_derivatives_names=['ATLAS'],
                              batch_size=1,
                              root_dir='./atlas/data/test/')
    mask = np.fft.fftshift(np.ones((dataset_dict['RES'], dataset_dict['RES']))).astype(np.complex64)

  for index in range(len(IDX_LIST)):
    if args.dataset in ['sphere', 'bandlimited']:
      idx = SEED_LIST[index]
    elif args.dataset == 'mri':
      idx = MRI_IDX_LIST[index]
    else:
      idx = IDX_LIST[index]
    print(f'Start id {idx}')
    if args.dataset == 'div2k':
      image_pil = Image.open(f'./DIV2K/DIV2K_train_LR_x8/0{str(idx)}x8.png')
    elif args.dataset == 'sphere':
      print(dataset_dict['bandlimit'])
      image = SparseSphereSignal(dimension=2, length=128, bandlimit=dataset_dict['bandlimit'], seed=idx, generate=False).signal
    elif args.dataset == 'bandlimited':
      print(dataset_dict['bandlimit'])
      image = BandlimitedSignal(dimension=2, length=128, bandlimit=dataset_dict['bandlimit'], seed=idx, generate=False).signal
    elif args.dataset == 'mri':
      RES = dataset_dict['RES']
      tmp = bids_loader.load_sample(idx = idx, data_only=True) / 255.0
      vol = resize(tmp, (1, RES, RES, RES))[0]
      rng = np.random.default_rng(args.seed)
      slice_idx = rng.integers(0, RES)
      image = vol[:, :, slice_idx]  # (RES, RES)
    if args.dataset not in ['sphere', 'bandlimited', 'mri']:
      image = np.array(image_pil.convert('L').resize((dataset_dict['RES'], dataset_dict['RES']))) / 255.0   
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

      hash_table_size, base_resolution = dataset_dict[args.param_to_run]['hash_table_size'], dataset_dict[args.param_to_run]['base_resolution']
      nums_levels = dataset_dict[args.param_to_run]['nums_levels']
      exp_dict = {}
      for num_levels in nums_levels:
        logger.info(f"Number of levels={num_levels}")
        max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
        eta, list_of_outputs, i = None, [], 0
        while abs(max_dist - prev_max_dist) > tol:
            model_config = (int(num_levels), hash_table_size, base_resolution, 2.0)
            output = fit_instant_ngp(image, model_config, iters=iters, learning_rate=learning_rate, log_interval = iters+1, seed=args.seed + i * num_levels, device=device, mask=mask)
            error = np.linalg.norm(image.flatten() - output['best_pred'].flatten())
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
            filename=f"../logs/2d_{args.dataset}/{MODEL_NAME}.log",  # Name of the log file
            level=logging.INFO,  # Minimum log level to capture
            format='%(message)s', # Log message format
            filemode='w', # Overwrite the file each time the script runs (use 'a' to append)
            force=True,
        )

        # Create a logger object
        logger = logging.getLogger(__name__)

        num_levels, base_resolution = dataset_dict[args.param_to_run]['num_levels'], dataset_dict[args.param_to_run]['base_resolution']
        hash_sizes = dataset_dict[args.param_to_run]['hash_sizes']

        exp_dict, tol = {}, 1e-6

        for hash_size in hash_sizes:
          logger.info(f"Hash Table Size={hash_size}")
          
          max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
          eta, list_of_outputs, i = None, [], 0

          while abs(max_dist - prev_max_dist) > tol:
              model_config = (num_levels, int(hash_size), base_resolution, 2.0)
              output = fit_instant_ngp(image, model_config, iters=iters, learning_rate=learning_rate, log_interval = iters+1, seed=args.seed + i * hash_size, device=device, mask=mask)
              error = np.linalg.norm(image.flatten() - output['best_pred'].flatten())
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

        with open(f"../results/2d_{args.dataset}/{MODEL_NAME}.pkl", "wb") as f:
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