import numpy as np
import scipy
import matplotlib.pyplot as plt
import numpy.fft as fft

import random
import torch
import os
from tqdm import tqdm
import csv
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#@title Synthetic Signal Generation Code 

#from https://github.com/voilalab/INR-benchmark/blob/main/bandlimited_signal.py

def bandlimit_filter(data, cutoff_low, cutoff_high):
    """
    bandlimiting filter with "circular" mask
    
    Args:
        data (np.ndarray): input data to be filtered
        cutoff_low (float): low frequency cutoff region (0 ~ 0.5, 0.5 is Nyquist freq.).
        cutoff_high (float): high frequency cutoff region (0 ~ 0.5, 0. is Nyquist freq.).
        
    Returns:
        np.ndarray: Bandlimited Data
    """
    # Discrete Fourier Transform 
    data_fft = fft.fftn(data)
    
    # make meshgrid in the frequency space
    frequencies = [fft.fftfreq(n, d=1.0) for n in data.shape]
    grid = np.meshgrid(*frequencies, indexing='ij')
    radius = np.sqrt(np.sum(np.array(grid) ** 2, axis=0))

    # Generate bandlimit mask
    mask = (cutoff_low/np.sqrt(2) <= radius) & (radius <= cutoff_high/np.sqrt(2))
    
    # apply filter using generated mask
    data_fft_filtered = data_fft * mask
    # vis_filter = np.log(np.abs(fft.fftshift(data_fft_filtered)))
    
    # reconstruct signal using ifft
    filtered_data = np.real(fft.ifftn(data_fft_filtered))
    
    return filtered_data
def generate_bandlimits(start, stop, num_points, base):
    """
    Generate non-linear spaced bandlimits.
    
    Args:
        start (float): Starting value of the bandlimit (e.g., 0.1).
        stop (float): Ending value of the bandlimit (e.g., 0.9).
        num_points (int): Number of bandlimits to generate.
        scale (str): 'log' for logarithmic spacing, 'exp' for exponential spacing.
    
    Returns:
        np.ndarray: Array of bandlimits.
    """
    
    
    # bandlimits = np.linspace(0, 1, num_points) ** 2 * (stop - start) + start
      # Adjust base for more or less steepness
    bandlimits = (np.logspace(0, 1, num_points, base=base) - 1) / (base - 1)
    bandlimits = bandlimits * (stop - start) + start

    return bandlimits


class BandlimitedSignal:
    """
    This signal class generates white noise and then filters it to have a desired maximum spatial frequency.
    """
    def __init__(self, dimension, length, bandlimit, seed= None, generate = True, super_resolution = False, sparse=False):
        self.class_name = self.__class__.__name__
        if generate:
            np.random.seed(seed)
            self.dimension = dimension
            # Make the length odd so that we can have symmetry around the zero frequency
            if length // 2 == length / 2:
                length = length + 1
            self.length = length  # per dimension length; assume same length signal in each dimension
            self.bandlimit = bandlimit  # as a fraction

            # Generate the signal as white noise
            dims = [self.length] * self.dimension
            self.signal = np.random.uniform(size=dims)
            # plt.figure()
            # plt.imshow(self.signal)
            # plt.savefig('signal.png')
            bandlimits = generate_bandlimits(0.0015, 0.7, 9, base = 300)
            # Filter the signal to have the desired bandwidth
            bandlimit_idx = int(str(bandlimit)[-1]) - 1
            bandlimit = bandlimits[bandlimit_idx]
            self.signal = bandlimit_filter(self.signal, 0, bandlimit)
            
            # plt.figure()
            # plt.imshow(self.signal)
            # plt.savefig('filtered_signal.png')
            # plt.figure()
            # plt.imshow(np.abs(filtered_dft))
            # plt.savefig('dft.png')
        else:
            self.signal = np.load(f'./dcr_error/target_signals/{self.class_name}/{seed}/{dimension}d_{self.class_name}_{bandlimit}_{seed}.npy')


class SparseSphereSignal:
    """
    This signal class generates random spheres of roughly constant total volume, with a desired radius analogous to bandwidth.
    Please note that generating 3D sphere signal takes a lot of time, I recommend you to use the generated signals.
    """
    def __init__(self, dimension, length, bandlimit, seed, occupied_fraction=0.1, generate = True, super_resolution = False, sparse=False):
        self.class_name = self.__class__.__name__
        if generate:
            np.random.seed(seed)
            # Initialize an empty signal
            self.length = length
            self.bandlimit = bandlimit
            self.dimension = dimension
            dims = [self.length] * self.dimension
            self.signal = np.zeros(dims)

            # Calculate sphere number and radius for this bandwidth
            occupied_cells = occupied_fraction * np.prod(dims)
            self.sphere_radius = self.length / (self.bandlimit * 100)
            sphere_volume = np.pi**(self.dimension / 2.0) * self.sphere_radius**self.dimension / scipy.special.gamma(1 + self.dimension / 2.0)
            self.num_spheres = int(occupied_cells / sphere_volume)

            # Generate spheres
            centers = np.random.uniform(low=0, high=self.length, size=(self.num_spheres, self.dimension))
            it = np.nditer(self.signal, flags=['multi_index'])
            for _ in tqdm(it):
                idx = it.multi_index
                # check if this cell is within the radius of any of the sphere centers
                for center in centers:
                    if np.linalg.norm(idx - center) <= self.sphere_radius:
                        self.signal[idx] = 1
                        break  # If this cell is inside any one sphere, we don't need to bother checking the other spheres
        else:
            self.signal = np.load(f'./target_signals/{self.class_name}/{seed}/{dimension}d_{self.class_name}_{bandlimit}_{seed}.npy')

class Voxel_Fitting:
    def __init__(self, dimension, length, bandlimit, seed, super_resolution = False, sparse = True):
        if not sparse:
            if not super_resolution:
                self.dir = './target_signals/dragon.npy'
            else: self.dir = './target_signals/dragon_sr.npy'
        else:
            if not super_resolution:
                self.dir = './target_signals/dragon_sparse.npy'
            else: self.dir = './target_signals/dragon_sr_sparse.npy'
        self.signal = np.array(np.load(self.dir), dtype = np.int32)

def set_seed(seed):
    tqdm.write(f"Selected Seed: {seed}")
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # if CUDA

if __name__ == '__main__':
    # signal = BandlimitedSignal(2, 100, 0.3)
    # class_list = [BandlimitedSignal]
    class_list = [SparseSphereSignal]

    dimensions = [3]
    target_dir = 'target_signals/'
    seed_list = [1234]
    # seed_list = [1234, 2026, 5678, 7890, 7618]
    

    for signal_class in class_list:
        signal_name = signal_class.__name__
        if not os.path.exists(target_dir+f'{signal_name}'):
            os.makedirs(target_dir+f'{signal_name}')
        for dimension in dimensions:
            if signal_name == "SparseSphereSignal" and dimension == 2:
                tqdm.write(f"{signal_name} {dimension}")
                # continue
            
            for seed in seed_list:
                if not os.path.exists(target_dir+f'{signal_name}/{seed}'):
                    os.makedirs(target_dir+f'{signal_name}/{seed}')
                fieldnames = ['seed', 'bandlimits', 'time']
                if not os.path.exists(target_dir+f'{signal_name}/{seed}'):
                    with open(target_dir+f'{signal_name}/{seed}/estimated_time_{seed}.csv', 'w', newline='') as f:
                        
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                set_seed(seed)
                # bandlimits = [0.1, 0.2, 0.3, 0.4, 0.5]

                # bandlimits = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                bandlimits = [0.3, 0.4]
                for bandlimit in tqdm(bandlimits, desc=f'{signal_name} {dimension}'):
                    if dimension == 3:
                        signal_length = 64
                        # continue
                    else:
                        signal_length = 128
                    tqdm.write(f"{signal_name} {dimension} {bandlimit}")
                   
                    signal = signal_class(dimension=dimension, length=signal_length, bandlimit=bandlimit, seed = seed, generate = True)
                    
                    tqdm.write(f'{signal_name} {dimension} {bandlimit}')
                    if dimension == 2:
                        plt.figure()
                        plt.imshow(signal.signal)
                        plt.savefig(f'{target_dir}{signal_name}/{seed}/{dimension}d_{signal_name}_{bandlimit}_{seed}.png')
                        plt.close()
                    np.save(f'{target_dir}{signal_name}/{seed}/{dimension}d_{signal_name}_{bandlimit}_{seed}.npy', signal.signal)
                    with open(target_dir+f'{signal_name}/{seed}/estimated_time_{seed}.csv', 'a', newline='') as f:
                        writer_iter = csv.DictWriter(f, fieldnames=fieldnames)
                        writer_iter.writerow({
                            'seed': seed, 
                            'bandlimits': bandlimit,
                        })

                

