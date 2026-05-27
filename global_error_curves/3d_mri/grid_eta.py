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
from bandlimited_signal import *


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--interp', default='sinc', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--gpu', default=0, type=int)
  args, _ = parser.parse_known_args()
  return args

PARAMS_DICT = {
    'grid_resos': [22, 26, 28, 30, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64],
    'RES': 64,
}

# IDX_LIST = [131, 456, 789, 101, 202]
IDX_LIST = [131]

MODEL_NAME = 'grid_eta'
bids_loader = bidsio.BIDSLoader(data_entities=[{'subject': '',
                                               'session': '',
                                               'suffix': 'T1w',
                                               'space': 'MNI152NLin2009aSym'}],
                                target_entities=[],
                                data_derivatives_names=['ATLAS'],
                                batch_size=1,
                                root_dir='./atlas/data/test/')



def run_exp(args):
  tol = 1e-6
  learning_rate, iters = 5e-2, 1000
  
  RES = PARAMS_DICT['RES']
  mask = np.fft.fftshift(np.ones((RES, RES, RES))).astype(np.complex64)
  device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'   

  for index in range(len(IDX_LIST)):
    idx = IDX_LIST[index]
    print(f'Start id {idx}')
    tmp = bids_loader.load_sample(idx = idx, data_only=True) / 255.0
    signal = resize(tmp, (1, RES, RES, RES))[0] 

    
    # 1. Configure the logger
    logging.basicConfig(
    filename=f"../logs/3d_mri/{MODEL_NAME}.log",  # Name of the log file
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
        logger.info(f"Grid Resolution={grid_reso}")
        max_dist, prev_max_dist, min_error, best_pred = float('inf'), -float('inf'), float('inf'), None
        eta, list_of_outputs, i = None, [], 0
        while abs(max_dist - prev_max_dist) > tol:
            output = fit_grid(target_signal=signal, r = grid_reso, iters = iters, lr=learning_rate, interp=args.interp, log_interval=iters+1, seed=args.seed + i * grid_reso, device=device, mask=mask)
            error = np.linalg.norm(signal.flatten() - output['best_pred'].flatten())
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

    with open(f"../results/3d_mri/{MODEL_NAME}.pkl", "wb") as f:
        pickle.dump(exp_dict, f)


    print(f'Done with id {idx}')
  # After the loop, you can close the file implicitly, 
  # or use logging.shutdown() to ensure all messages are written.
  logging.shutdown() 

if __name__ == "__main__":
    args = parse_args()
    run_exp(args)
