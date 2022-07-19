#!/usr/bin/env python

import numpy as np
from skimage.metrics import structural_similarity as ssim_func

import torch

import utils

class TVNorm():
    def __init__(self, mode='l1'):
        self.mode = mode
    def __call__(self, img):
        grad_x = img[..., 1:, 1:] - img[..., 1:, :-1]
        grad_y = img[..., 1:, 1:] - img[..., :-1, 1:]
        
        if self.mode == 'isotropic':
            return torch.sqrt(grad_x**2 + grad_y**2).mean()
        elif self.mode == 'l1':
            return abs(grad_x).mean() + abs(grad_y).mean()
        else:
            return (grad_x.pow(2) + grad_y.pow(2)).mean()     
    
class HessianNorm():
    def __init__(self):
        pass
    def __call__(self, img):
        # Compute Individual derivatives
        fxx = img[..., 1:-1, :-2] + img[..., 1:-1, 2:] - 2*img[..., 1:-1, 1:-1]
        fyy = img[..., :-2, 1:-1] + img[..., 2:, 1:-1] - 2*img[..., 1:-1, 1:-1]
        fxy = img[..., :-1, :-1] + img[..., 1:, 1:] - \
              img[..., 1:, :-1] - img[..., :-1, 1:]
          
        return torch.sqrt(fxx.abs().pow(2) +\
                          2*fxy[..., :-1, :-1].abs().pow(2) +\
                          fyy.abs().pow(2)).mean()
    
class L1Norm():
    def __init__(self):
        pass
    def __call__(self, x):
        return abs(x).mean()        

class L2Norm():
    def __init__(self):
        pass
    def __call__(self, x1, x2):
        return ((x1-x2).pow(2)).mean()    
 
@torch.no_grad()
def get_metrics(gt, estim, lpip_func=None, pad=True):
    '''
        Compute SNR, PSNR, SSIM, and LPIP between two images.
        
        Inputs:
            gt: Ground truth image
            estim: Estimated image
            lpip_func: CUDA function for computing lpip value
            pad: if True, remove boundaries when computing metrics
            
        Outputs:
            metrics: dictionary with following fields:
                snrval: SNR of reconstruction
                psnrval: Peak SNR 
                ssimval: SSIM
                lpipval: VGG perceptual metrics
    '''
    if min(gt.shape) < 50:
        pad = False
    if pad:
        gt = gt[20:-20, 20:-20]
        estim = estim[20:-20, 20:-20]
        
    snrval = utils.asnr(gt, estim)
    psnrval = utils.asnr(gt, estim, compute_psnr=True)
    ssimval = ssim_func(gt, estim, multichannel=True)
    
    # Need to convert images to tensors for computing lpips
    gt_ten = torch.tensor(gt)[None, None, ...]
    estim_ten = torch.tensor(estim)[None, None, ...]
    
    # For some reason, the image values should be [-1, 1]
    if lpip_func is not None:
        lpipval = lpip_func(2*gt_ten-1, 2*estim_ten-1).item()
    else:
        lpipval = 0
    
    metrics = {'snrval': snrval,
               'psnrval': psnrval,
               'ssimval': ssimval,
               'lpipval': lpipval}
    return metrics