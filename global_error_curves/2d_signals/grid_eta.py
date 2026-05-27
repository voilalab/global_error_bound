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
from fns_grid import *
from bandlimited_signal import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', default='div2k', type=str)
  parser.add_argument('--interp', default='sinc', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  args, _ = parser.parse_known_args()
  return args

PARAMS_DICT = {
  'div2k': {
    'grid_resos': [28, 40, 48, 56, 64, 70, 74, 80, 84, 90, 94, 98, 102, 106, 110, 114, 118, 120, 124, 128],
    'RES': 128,
  },
  'sphere': {
    'grid_resos': [28, 40, 48, 56, 64, 70, 74, 80, 84, 90, 94, 98, 102, 106, 110, 114, 118, 120, 124, 128],
    'bandlimit': 0.3,
  },
  'bandlimited': {
    'grid_resos': [28, 40, 48, 56, 64, 70, 74, 80, 84, 90, 94, 98, 102, 106, 110, 114, 118, 120, 124, 128],
    'bandlimit': 0.9,
  },
  'mri': {
    'grid_resos': [28, 40, 48, 56, 64, 70, 74, 80, 84, 90, 94, 98, 102, 106, 110, 114, 118, 120, 124, 128],
    'RES': 128,
  },
}

# IDX_LIST = [131, 456, 789, 101, 202]
# SEED_LIST = [1234, 2026, 5678, 7890, 7618]
# MRI_IDX_LIST =  [131, 56, 89, 101, 202]
IDX_LIST = [131]
SEED_LIST = [1234]
MRI_IDX_LIST =  [131]

MODEL_NAME = 'grid_eta'

def run_exp(args):
  tol = 1e-6
  learning_rate, iters = 5e-2, 1000
  
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

    grid_resos = dataset_dict['grid_resos']
    exp_dict = {}
    torch.manual_seed(args.seed)

    for grid_reso in grid_resos:
        logger.info(f"Grid Resolution={grid_reso}")
        max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
        eta, list_of_outputs, i = None, [], 0
        while abs(max_dist - prev_max_dist) > tol:
            output = fit_grid(target_signal=image, r = grid_reso, iters = iters, lr=learning_rate, interp=args.interp, log_interval=iters+1, seed=args.seed + i * grid_reso, device=device, mask=mask)
            error = np.linalg.norm(image.flatten() - output['best_pred'].flatten())
            list_of_outputs.append(output['best_pred'].flatten())
            i+=1
            if i > 5:
              if len(list_of_outputs) >= 2:
                  prev_max_dist = max_dist
                  max_dist, eta = max_pairwise_l2_distance(list_of_outputs)
              if error < min_error:
                  min_error, best_pred = error, output['best_pred'].flatten()
              logger.info(f"    Current and previous max pairwise L2 distance={max_dist} and {prev_max_dist}, min error={min_error} with loss={output['best_loss']:.3e}")
            else:
              logger.info(f"   Iteration {i} with loss={output['best_loss']:.3e} and error={error}")
            
        logger.info(f"Final eta for grid resolution {grid_reso} is {max_dist}")
        exp_dict[f'{grid_reso}'] = {}
        exp_dict[f'{grid_reso}']['eta'] = eta
        exp_dict[f'{grid_reso}']['eta_upper_bound'] = max_dist
        exp_dict[f'{grid_reso}']['error'] = min_error
        exp_dict[f'{grid_reso}']['pred'] = best_pred
        logger.info("-----------------------------------------------------------")

    with open(f"../results/2d_{args.dataset}/{MODEL_NAME}.pkl", "wb") as f:
        pickle.dump(exp_dict, f)

    print(f'Done with id {idx}')
  # After the loop, you can close the file implicitly, 
  # or use logging.shutdown() to ensure all messages are written.
  logging.shutdown() 

if __name__ == "__main__":
    args = parse_args()
    run_exp(args)
