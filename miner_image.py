#!/usr/bin/env python

import sys
import importlib
import time
import os

import numpy as np
from scipy import io
from skimage.metrics import structural_similarity as ssim_func

import cv2
import torch

import matplotlib.pyplot as plt
from matplotlib import cm
plt.gray()

sys.path.append('modules')

import utils
import miner

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    expname = 'pluto'                       # Image name (jpg)
    configname = 'configs/image_32x32.ini'  # Configuration file
    scale = 0.25             # Set scale to 0.5 for faster results
    stopping_mse = 1e-4     # When to stop image fitting
    target_mse = 1e-4   # Per-block stopping criterion
    nscales = 4                 # Number of MINER scales
    
    # Read configuration
    config = miner.load_config(configname)
    config.signaltype = 'image'

    # Load image and scale
    im = cv2.imread('data/%s.jpg'%expname).astype(np.float32)/255.0
    im = cv2.resize(im, None, fx=scale, fy=scale)
    # Clipping image ensures there are a whole number of blocks
    clip_size = (config.stride*pow(2, nscales-1),
                 config.stride*pow(2, nscales-1))
    im = utils.moduloclip(im, clip_size)
    H, W, _ = im.shape
    
    config.savedir = '../results/%s_%d'%(expname, H)
    os.makedirs(config.savedir, exist_ok=True)
    
    # Now run MINER
    tic = time.time()
    best_im, info = miner.miner_imfit(im, nscales,
                                        target_mse,
                                        stopping_mse,
                                        config)
    total_time = time.time() - tic
    nparams = info['nparams']

    im_labels = miner.drawblocks(info['learn_indices_list'],
                                im.shape[:2], config.ksize)
    im_labels = im_labels.numpy()
            

    
    print('Total time %.2f minutes'%(total_time/60))
    print('PSNR: ', utils.psnr(im, best_im))
    print('SSIM: ', ssim_func(im, best_im, multichannel=True))
    print('Total pararmeters: %.2f million'%(nparams/1e6))    
    
    time_array = info['time_array']
    time_array = time_array - time_array[0]
    mse_array = info['mse_array']
    
    os.makedirs('results', exist_ok=True)
    mdict = {'mse_array': mse_array,
             'time_array': time_array,
             'nparams': nparams,
             'memory_array': info['memory_array'],
             'nparams_array': info['nparams_array']}
    
    io.savemat('results/%s_miner.mat'%expname, mdict)
    
    plt.subplot(1, 2, 1)
    plt.semilogy(time_array, mse_array)
    plt.xlabel('Time (s)')
    plt.ylabel('MSE')
    plt.grid()
    
    im_labels_colored = cm.hsv(im_labels/nscales)[..., :3]
    im_labels_colored *= (im_labels[..., np.newaxis] > 0)
    
    im_marked = im[..., ::-1]*(im_labels[..., np.newaxis] == 0)
    
    im_marked += im_labels_colored
    
    plt.subplot(1, 2, 2)
    plt.imshow(im_marked)
    plt.show()
