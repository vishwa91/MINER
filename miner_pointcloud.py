#!/usr/bin/env python

import os
import sys
import time

import numpy as np
from scipy import io
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim_func

import matplotlib.pyplot as plt
plt.gray()

sys.path.append('modules')

import miner
import volutils

# NOTE: Although we are training on an binary occupancy volume, we found an L2
# loss to be dramatically faster than logistic loss.
if __name__ == '__main__':
    expname = 'lucy'            # Example file with uniform sampling
    configname = 'configs/volume_16x16x16.ini'  # Configuration file
    target_mse = 2e-4           # Per block stopping threshold
    stopping_iou = 0.99         # What IoU to stop at
    scale = 0.5                 # Set to a smaller value for faster results
    nscales = 4                 # Number of miner scales
    
    config = miner.load_config(configname)

    # Load the data. Uniformly sampled for fast loading and processing
    im = io.loadmat('data/%s.mat'%expname)['hypercube'].astype(np.float32)
    
    config.signaltype = 'occupancy'
    
    # This threshold is for marching cubes.
    config.mcubes_thres = 0.4

    # Load image and scale
    im = ndimage.zoom(im/im.max(), [scale, scale, scale], order=0)
        
    H, W, T = im.shape
    config.savedir = 'results/%s'%expname
    
    os.makedirs(config.savedir, exist_ok=True)
    
    best_im = np.zeros((H, W, T), dtype=np.float32)
    
    # Now run MINER on volumetric data
    tic = time.time()
    best_im, info = miner.miner_volfit(im, nscales,
                                        target_mse,
                                        stopping_iou,
                                        config)
    nparams = info['nparams']
    nparams_array = info['nparams_array']
    time_epoch_array = info['time_epoch_array']
    time_array = info['time_array']
    
    mse_array = info['mse_array']
    
    total_time = time.time() - tic
    
    os.makedirs('results', exist_ok=True)
    mdict = {'mse_array': mse_array,
             'time_array': time_array-time_array[0],
             'total_time': total_time,
             'mse_epoch_array': info['mse_epoch_array'],
             'mem_array': info['mem_array'],
             'time_epoch_array': time_epoch_array,
             'nparams_array': nparams_array,
             'nparams': nparams,
             'nparams_array': info['nparams_array']}
    
    learn_indices = info['learn_indices_list']
    for idx in range(nscales):
        mdict['learn_indices%d'%idx] = learn_indices[idx].numpy()
    mdict['cubesize'] = np.array([H, W, T])
    mdict['ksize'] = config.ksize
    
    
    io.savemat('results/%s_miner.mat'%expname, mdict)
    
    volutils.march_and_save(best_im, config.mcubes_thres,
                            'results/%s_miner_volume.dae'%expname,
                            True)
    
    print('Total time %.2f minutes'%(total_time/60))
    print('IoU: ', volutils.get_IoU(best_im, im, config.mcubes_thres))
    print('Total pararmeters: %.2f million'%(nparams/1e6))
    
    plt.loglog(time_array-time_array[0], mse_array)
    plt.xlabel('Time (s)')
    plt.ylabel('MSE')
    plt.grid()
    plt.show()
