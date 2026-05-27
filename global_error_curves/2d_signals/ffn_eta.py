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
from fns_ffn import *
from bandlimited_signal import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='div2k', type=str)
  parser.add_argument('--param_to_run', default='width', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  args, _ = parser.parse_known_args()
  return args

PARAMS_DICT = {
  'div2k': {
    'mapping_size': {
      'mapping_sizes': np.arange(10, 211, 1),
      'num_layers': 1,
      'width': 64,
      'threshold': 50,
    },
    'width': {
      'widths': np.arange(1, 65, 1),
      'mapping_size': 256,
      'num_layers': 1,
      'threshold': 10,
    },
    'depth': {
      'nums_layers': np.arange(1, 8, 1),
      'mapping_size': 256,
      'width': 32,
    },
    'RES': 128,
  },
  'sphere': {
    'mapping_size': {
      'mapping_sizes': np.arange(10, 221, 1),
      'num_layers': 1,
      'width': 64,
      'threshold': 50,
    },
    'width': {
      'widths': np.arange(1, 65, 1),
      'mapping_size': 256,
      'num_layers': 1,
      'threshold': 10,
    },
    'depth': {
      'nums_layers': np.arange(1, 8, 1),
      'mapping_size': 256,
      'width': 32,
    },
    'bandlimit': 0.3,
  },
  'bandlimited': {
    'mapping_size': {
      'mapping_sizes': np.arange(10, 221, 1),
      'num_layers': 1,
      'width': 64,
      'threshold': 50,
    },
    'width': {
      'widths': np.arange(1, 65, 1),
      'mapping_size': 256,
      'num_layers': 1,
      'threshold': 10,
    },
    'depth': {
      'nums_layers': np.arange(1, 8, 1),
      'mapping_size': 256,
      'width': 32,
    },
    'bandlimit': 0.9,
  },
  'mri': {
    'mapping_size': {
      'mapping_sizes': np.arange(10, 211, 1),
      'num_layers': 1,
      'width': 64,
      'threshold': 30,
    },
    'width': {
      'widths': np.arange(1, 65, 1),
      'mapping_size': 256,
      'num_layers': 1,
      'threshold': 10,
    },
    'depth': {
      'nums_layers': np.arange(1, 8, 1),
      'mapping_size': 256,
      'width': 32,
    },
    'RES': 128,
  },
}

IDX_LIST = [131, 456, 789, 101, 202]
# SEED_LIST = [1234, 2026, 5678, 7890, 7618]
MRI_IDX_LIST =  [131, 56, 89, 101, 202]
SEED_LIST = [1234]

MODEL_NAME = 'ffn_eta'

def run_exp(args):
  scale, tol = 20., 1e-6
  learning_rate, iters = 1e-3, 8000
  
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
    if args.param_to_run == 'mapping_size':
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

      num_layers, width = dataset_dict[args.param_to_run]['num_layers'], dataset_dict[args.param_to_run]['width']
      mapping_sizes = dataset_dict[args.param_to_run]['mapping_sizes']
      exp_dict = {}
      torch.manual_seed(args.seed)
      B = torch.randn(mapping_sizes[-1], len(image.shape), dtype=torch.float32, device=device) * scale

      for mapping_size in mapping_sizes:
          logger.info(f"mapping_size={mapping_size}")
          max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
          eta, list_of_outputs, i = None, [], 0
          while abs(max_dist - prev_max_dist) > tol:
              output = fit_fourier_features_learn(y_gt=image, B=B[:mapping_size, :], network_size=(num_layers, width), iters=iters,
                                                                    learning_rate=learning_rate,log_interval=iters+1, seed=args.seed + i * mapping_size, device=device, mask=mask)
              error = np.linalg.norm(image.flatten() - output['best_pred'].flatten())
              list_of_outputs.append(output['best_pred'].flatten())
              i+=1
              if mapping_size > dataset_dict[args.param_to_run]['threshold'] or i > 9:
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

      with open(f"pkls/{MODEL_NAME}/{args.dataset}/{args.param_to_run}_i{idx}_s{args.seed}.pkl", "wb") as f:
          pickle.dump(exp_dict, f)

    elif args.param_to_run == 'width':
        # 1. Configure the logger
        logging.basicConfig(
            filename=f"../results/2d_{args.dataset}/{MODEL_NAME}.log",  # Name of the log file
            level=logging.INFO,  # Minimum log level to capture
            format='%(message)s', # Log message format
            filemode='w', # Overwrite the file each time the script runs (use 'a' to append)
            force=True,
        )

        # Create a logger object
        logger = logging.getLogger(__name__)

        num_layers, mapping_size = dataset_dict[args.param_to_run]['num_layers'], dataset_dict[args.param_to_run]['mapping_size']
        widths = dataset_dict[args.param_to_run]['widths']

        exp_dict, tol = {}, 1e-6

        torch.manual_seed(args.seed)
        B = torch.randn(mapping_size, len(image.shape), dtype=torch.float32, device=device) * scale
        for width in widths:

            logger.info(f"width={width}")
            
            max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
            eta, list_of_outputs, i = None, [], 0
            while abs(max_dist - prev_max_dist) > tol:
                output = fit_fourier_features_learn(y_gt=image, B=B, network_size=(num_layers, width), iters=iters,
                                                                      learning_rate=learning_rate,
                                                                      log_interval=iters+1, seed=args.seed + i * width, device=device, mask=mask)
                error = np.linalg.norm(image.flatten() - output['best_pred'].flatten())
                list_of_outputs.append(output['best_pred'].flatten())
                i+=1
                if width > dataset_dict[args.param_to_run]['threshold'] or i > 9:
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

        with open(f"../results/2d_{args.dataset}/{MODEL_NAME}.pkl", "wb") as f:
            pickle.dump(exp_dict, f)

    elif args.param_to_run == 'depth':
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

        width, mapping_size = dataset_dict[args.param_to_run]['width'], dataset_dict[args.param_to_run]['mapping_size']
        nums_layers = dataset_dict[args.param_to_run]['nums_layers']

        exp_dict, tol = {}, 1e-6

        torch.manual_seed(args.seed)
        B = torch.randn(mapping_size, len(image.shape), dtype=torch.float32, device=device) * scale
        for num_layers in nums_layers:

            logger.info(f"Number of hidden layers={num_layers}")
            
            max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
            eta, list_of_outputs, i = None, [], 0
            while abs(max_dist - prev_max_dist) > tol:
                output = fit_fourier_features_learn(y_gt=image, B=B, network_size=(num_layers, width), iters=iters,
                                                                      learning_rate=learning_rate,
                                                                      log_interval=iters+1, seed=args.seed + i * num_layers, device=device, mask=mask)
                error = np.linalg.norm(image.flatten() - output['best_pred'].flatten())
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

        with open(f"pkls/{MODEL_NAME}/{args.dataset}/{args.param_to_run}_i{idx}_s{args.seed}.pkl", "wb") as f:
            pickle.dump(exp_dict, f)
    else:
      print(f"Parameter {args.param_to_run} not recognized. Skipping.")

    print(f'Done with id {idx}')
  # After the loop, you can close the file implicitly, 
  # or use logging.shutdown() to ensure all messages are written.
  logging.shutdown() 

if __name__ == "__main__":
    args = parse_args()
    run_exp(args, dataset)
