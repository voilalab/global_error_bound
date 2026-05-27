import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets
import pickle
from PIL import Image
import logging
import bidsio
import argparse
import sys
sys.path.append('../../')
from helpers import *
from fns_gap import *
from bandlimited_signal import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='div2k', type=str)
  parser.add_argument('--param_to_run', default='feature_dim', type=str)
  parser.add_argument('--operation', default='concatenate', type=str)
  parser.add_argument('--decoder', default='linear', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  args, _ = parser.parse_known_args()
  return args

PARAMS_DICT = {
  'div2k': {
    'plane_resolution': {
      'plane_resos': np.arange(1, 129, 1),
      'lineres': 4,
      'feature_dim': 4,
      'hidden_dim': 32,
    },
    'feature_dim': {
      'feature_dims': np.arange(10, 257, 1),
      'lineres': 128,
      'planeres': 2,
      'hidden_dim': 128,
    },
    'RES': 128,
  },
  'sphere': {
    'plane_resolution': {
      'plane_resos': np.arange(1, 129, 1),
      'lineres': 4,
      'feature_dim': 4,
      'hidden_dim': 32,
    },
    'feature_dim': {
      'feature_dims': np.arange(10, 257, 1),
      'lineres': 128,
      'planeres': 2,
      'hidden_dim': 128,
    },
    'bandlimit': 0.3,
  },
  'bandlimited': {
    'plane_resolution': {
      'plane_resos': np.arange(1, 129, 1),
      'lineres': 4,
      'feature_dim': 4,
      'hidden_dim': 32,
    },
    'feature_dim': {
      'feature_dims': np.arange(10, 257, 1),
      'lineres': 128,
      'planeres': 2,
      'hidden_dim': 128,
    },
    'bandlimit': 0.9,
  },
  'mri': {
    'plane_resolution': {
      'plane_resos': np.arange(1, 129, 1),
      'lineres': 4,
      'feature_dim': 4,
      'hidden_dim': 32,
    },
    'feature_dim': {
      'feature_dims': np.arange(10, 257, 1),
      'lineres': 128,
      'planeres': 2,
      'hidden_dim': 128,
    },
    'RES': 128,
  },
}

IDX_LIST = [131, 456, 789, 101, 202]
# SEED_LIST = [1234, 2026, 5678, 7890, 7618]
SEED_LIST = [1234]
MRI_IDX_LIST =  [131, 56, 89, 101, 202]

MODEL_NAME = 'ga-planes_eta'

def run_exp(args):
  tol = 1e-6
  learning_rate, iters = 5e-3, 4000
  
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

    # Set idx based on signal type
    if args.dataset in ['sphere', 'bandlimited']:
      idx = SEED_LIST[index]
    elif args.dataset == 'mri':
      idx = MRI_IDX_LIST[index]
    else:
      idx = IDX_LIST[index]
    print(f'Start id {idx}')

    # Initialize signal based on signal type
    if args.dataset == 'div2k':
      image_pil = Image.open(f'./DIV2K/DIV2K_train_LR_x8/0{str(idx)}x8.png')
    elif args.dataset == 'sphere':
      image = SparseSphereSignal(dimension=2, length=128, bandlimit=dataset_dict['bandlimit'], seed=idx, generate=False).signal
    elif args.dataset == 'bandlimited':
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

    if args.param_to_run == 'plane_resolution':
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
        "lineres": dataset_dict[args.param_to_run]['lineres'],
        "line_feature_dim": dataset_dict[args.param_to_run]['feature_dim'],
        "plane_feature_dim": 1,
        "hidden_dim": dataset_dict[args.param_to_run]['hidden_dim'],
        "operation": args.operation,
        "decoder": args.decoder,
      }

      plane_resos, exp_dict = dataset_dict[args.param_to_run]['plane_resos'], {}
      for plane_reso in plane_resos:
        logger.info(f"Plane resolution={plane_reso}")
        gap_args['planeres'] = plane_reso
        max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
        eta, list_of_outputs, i = None, [], 0
        while abs(max_dist - prev_max_dist) > tol:
            output = fit_gaplanes(args=gap_args, img=image, iters=iters, learning_rate=learning_rate, log_interval = iters+1, seed=args.seed + i * plane_reso, device=device, mask=mask)
            error = np.linalg.norm(image.flatten() - output['best_pred'].flatten())
            list_of_outputs.append(output['best_pred'].flatten())
            if len(list_of_outputs) >= 2:
                prev_max_dist = max_dist
                max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
            if error < min_error:
                min_error, best_pred = error, output['best_pred'].flatten()
            logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
            i+=1
        logger.info(f"Final eta for plane resolution {plane_reso} is {max_dist}")
        exp_dict[f'{plane_reso}'] = {}
        exp_dict[f'{plane_reso}']['eta'] = eta
        exp_dict[f'{plane_reso}']['eta_upper_bound'] = max_dist
        exp_dict[f'{plane_reso}']['error'] = min_error
        exp_dict[f'{plane_reso}']['pred'] = best_pred
        logger.info("-----------------------------------------------------------")

      with open(f"pkls/{MODEL_NAME}/{args.dataset}/{args.param_to_run}_i{idx}_s{args.seed}.pkl", "wb") as f:
          pickle.dump(exp_dict, f)
    elif args.param_to_run == 'feature_dim':
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

      gap_args = {
        "lineres": dataset_dict[args.param_to_run]['lineres'],
        "planeres": dataset_dict[args.param_to_run]['planeres'],
        "plane_feature_dim": 1,
        "hidden_dim": dataset_dict[args.param_to_run]['hidden_dim'],
        "operation": args.operation,
        "decoder": args.decoder,
      }

      feature_dims, exp_dict = dataset_dict[args.param_to_run]['feature_dims'], {}
      for feature_dim in feature_dims:
        logger.info(f"Feature Dimension={feature_dim}")
        gap_args['line_feature_dim'] = feature_dim
        max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
        eta, list_of_outputs, i = None, [], 0
        while abs(max_dist - prev_max_dist) > tol:
            output = fit_gaplanes(args=gap_args, img=image, iters=iters, learning_rate=learning_rate, log_interval = iters+1, seed=args.seed + i * feature_dim, device=device, mask=mask)
            error = np.linalg.norm(image.flatten() - output['best_pred'].flatten())
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
