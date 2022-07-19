#!/usr/bin/env python

'''
    Miscellaneous utilities that are extremely helpful but cannot be clubbed
    into other modules.
'''

# System imports
import os
import sys
import time
import pickle
import pdb
import glob
import h5py

import torch

# Scientific computing
import numpy as np
import scipy as sp
import scipy.linalg as lin
import scipy.ndimage as ndim
from scipy import io
from scipy import signal
from scipy.sparse import linalg

# Plotting
import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def load_raw(filename, nrows, ncols, dtype=np.uint16):
    '''
        Load a RAW image from a file

        Inputs:
            filename: Name of the file to load. Should have an
                extension of '.Raw'
            nrows, ncols: Number of rows and columns in the image
            dtype: Datatype of the file

        Outputs:
            im: Loaded image
    '''
    with open(filename, 'rb') as fd:
        rawdata = np.fromfile(fd, dtype=dtype, count=nrows*ncols)

    im = rawdata.reshape(nrows, ncols)

    return im

def stack2mosaic(imstack):
    '''
        Convert a 3D stack of images to a 2D mosaic

        Inputs:
            imstack: (H, W, nimg) stack of images

        Outputs:
            immosaic: A 2D mosaic of images
    '''
    H, W, nimg = imstack.shape

    nrows = int(np.ceil(np.sqrt(nimg)))
    ncols = int(np.ceil(nimg/nrows))

    immosaic = np.zeros((H*nrows, W*ncols), dtype=imstack.dtype)

    for row_idx in range(nrows):
        for col_idx in range(ncols):
            img_idx = row_idx*ncols + col_idx
            if img_idx >= nimg:
                return immosaic

            immosaic[row_idx*H:(row_idx+1)*H, col_idx*W:(col_idx+1)*W] = \
                                              imstack[:, :, img_idx]

    return immosaic

def nextpow2(x):
    '''
        Return smallest number larger than x and a power of 2.
    '''
    logx = np.ceil(np.log2(x))
    return pow(2, logx)

def normalize(x, fullnormalize=False):
    '''
        Normalize input to lie between 0, 1.

        Inputs:
            x: Input signal
            fullnormalize: If True, normalize such that minimum is 0 and
                maximum is 1. Else, normalize such that maximum is 1 alone.

        Outputs:
            xnormalized: Normalized x.
    '''

    if x.sum() == 0:
        return x
    
    xmax = x.max()

    if fullnormalize:
        xmin = x.min()
    else:
        xmin = 0

    xnormalized = (x - xmin)/(xmax - xmin)

    return xnormalized

def asnr(x, xhat, compute_psnr=False):
    '''
        Compute affine SNR, which accounts for any scaling and shift between two
        signals

        Inputs:
            x: Ground truth signal(ndarray)
            xhat: Approximation of x

        Outputs:
            asnr_val: 20log10(||x||/||x - (a.xhat + b)||)
                where a, b are scalars that miminize MSE between x and xhat
    '''
    mxy = (x*xhat).mean()
    mxx = (xhat*xhat).mean()
    mx = xhat.mean()
    my = x.mean()
    

    a = (mxy - mx*my)/(mxx - mx*mx)
    b = my - a*mx

    if compute_psnr:
        return psnr(x, a*xhat + b)
    else:
        return rsnr(x, a*xhat + b)

def rsnr(x, xhat):
    '''
        Compute reconstruction SNR for a given signal and its reconstruction.

        Inputs:
            x: Ground truth signal (ndarray)
            xhat: Approximation of x

        Outputs:
            rsnr_val: RSNR = 20log10(||x||/||x-xhat||)
    '''
    xn = lin.norm(x.reshape(-1))
    en = lin.norm((x-xhat).reshape(-1)) + 1e-12
    rsnr_val = 20*np.log10(xn/en)

    return rsnr_val

def psnr(x, xhat):
    ''' Compute Peak Signal to Noise Ratio in dB

        Inputs:
            x: Ground truth signal
            xhat: Reconstructed signal

        Outputs:
            snrval: PSNR in dB
    '''
    err = x - xhat
    denom = np.mean(pow(err, 2)) + 1e-12

    snrval = 10*np.log10(np.max(x)/denom)

    return snrval

def SAM_3d(x, xhat, avg=False):
    '''
        Compute SAM for a 3D hyperspectral cube

        Inputs:
            x: Ground truth HSI
            xhat: Reconstructed HSI
            avg: If True, average spatially

        Outputs:
            SAM: SAM map (or average value)
    '''

    x_norm = (x*x).sum(2)
    xhat_norm = (xhat*xhat).sum(2)

    xxhat = abs(x*xhat).sum(2)

    SAM = xxhat/np.sqrt(x_norm*xhat_norm + 1e-12)

    if avg:
        SAM = np.mean(SAM)

    return SAM

def savep(data, filename):
    '''
        Tiny wrapper to store data as a python pickle.

        Inputs:
            data: List of data
            filename: Name of file to save
    '''
    f = open(filename, 'wb')
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def loadp(filename):
    '''
        Tiny wrapper to load data from python pickle.

        Inputs:
            filename: Name of file to load from

        Outputs:
            data: Output data from pickle file
    '''
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()

    return data

def dither(im):
    '''
        Implements Floyd-Steinberg spatial dithering algorithm

        Inputs:
            im: Grayscale image normalized between 0, 1

        Outputs:
            imdither: Dithered image
    '''
    H, W = im.shape
    imdither = np.zeros((H+1, W+1))

    # Pad the last row/column to propagate error
    imdither[:H, :W] = im
    imdither[H, :W] = im[H-1, :W]
    imdither[:H, W] = im[:H, W-1]
    imdither[H, W] = im[H-1, W-1]

    for h in range(0, H):
        for w in range(1, W):
            oldpixel = imdither[h, w]
            newpixel = (oldpixel > 0.5)
            imdither[h, w] = newpixel

            err = oldpixel - newpixel
            imdither[h, w+1] += (err * 7.0/16)
            imdither[h+1, w-1] += (err * 3.0/16)
            imdither[h+1, w] += (err * 5.0/16)
            imdither[h+1, w+1] += (err * 1.0/16)

    return imdither[:H, :W]

def embed(im, embedsize):
    '''
        Embed a small image centrally into a larger window.

        Inputs:
            im: Image to embed
            embedsize: 2-tuple of window size

        Outputs:
            imembed: Embedded image
    '''

    Hi, Wi = im.shape
    He, We = embedsize

    dH = (He - Hi)//2
    dW = (We - Wi)//2

    imembed = np.zeros((He, We), dtype=im.dtype)
    imembed[dH:Hi+dH, dW:Wi+dW] = im

    return imembed

def measure(x, noise_snr=40, tau=100):
    ''' Realistic sensor measurement with readout and photon noise

        Inputs:
            noise_snr: Readout noise in electron count
            tau: Integration time. Poisson noise is created for x*tau.
                (Default is 100)

        Outputs:
            x_meas: x with added noise
    '''
    x_meas = np.copy(x)

    #noise = pow(10, -noise_snr/20)*np.random.randn(x_meas.size).reshape(x_meas.shape)
    noise = np.random.randn(x_meas.size).reshape(x_meas.shape)*noise_snr

    # First add photon noise, provided it is not infinity
    if tau != float('Inf'):
        x_meas = x_meas*tau

        x_meas[x > 0] = np.random.poisson(x_meas[x > 0])
        x_meas[x <= 0] = -np.random.poisson(-x_meas[x <= 0])

        x_meas = (x_meas + noise)/tau

    else:
        x_meas = x_meas + noise

    return x_meas

def rician(sig, noise_snr):
    '''
        Add Rician noise
        
        Inputs:
            sig: N dimensional signal
            noise_snr: std. dev of input noise
            
        Outputs:
            sig_rician: Rician corrupted signal
    '''
    n1 = np.random.randn(*sig.shape)*noise_snr
    n2 = np.random.randn(*sig.shape)*noise_snr

    return np.sqrt((sig + n1)**2 + n2**2)

def deconvwnr1(sig, kernel, wconst=1e-2):
    '''
        Deconvolve a 1D signal using Wiener deconvolution

        Inputs:
            sig: Input signal
            kernel: Impulse response
            wconst: Wiener deconvolution constant

        Outputs:
            sig_deconv: Deconvolved signal
    '''

    sigshape = sig.shape
    sig = sig.ravel()
    kernel = kernel.ravel()

    N = sig.size
    M = kernel.size

    # Padd signal to regularize 
    sig_padded = np.zeros(N+2*M)
    sig_padded[M:-M] = sig

    # Compute Fourier transform
    sig_fft = np.fft.fft(sig_padded)
    kernel_fft = np.fft.fft(kernel, n=(N+2*M))

    # Compute inverse kernel
    kernel_inv_fft = np.conj(kernel_fft)/(np.abs(kernel_fft)**2 + wconst)

    # Now compute deconvolution
    sig_deconv_fft = sig_fft*kernel_inv_fft

    # Compute inverse fourier transform
    sig_deconv_padded = np.fft.ifft(sig_deconv_fft)

    # Clip
    sig_deconv = np.real(sig_deconv_padded[M//2:M//2+N])

    return sig_deconv.reshape(sigshape)

def lowpassfilter(data, order=5, freq=0.5):
    '''
        Low pass filter the input data with butterworth filter.
        This is based on Zackory's github repo: 
            https://github.com/Healthcare-Robotics/smm50

        Inputs:
            data: Data to be filtered with each row being a spectral profile
            order: Order of butterworth filter
            freq: Cutoff frequency

        Outputs:
            data_smooth: Smoothed spectral profiles
    '''
    # Get butterworth coefficients
    b, a = signal.butter(order, freq, analog=False)

    # Then just apply the filter
    data_smooth = signal.filtfilt(b, a, data)

    return data_smooth

def grid_plot(imdata):
    '''
        Plot 3D set of images into a 2D grid using subplots.

        Inputs:
            imdata: N x H x W image stack

        Outputs:
            None
    '''
    N, H, W = imdata.shape

    nrows = int(np.sqrt(N))
    ncols = int(np.ceil(N/nrows))

    for idx in range(N):
        plt.subplot(nrows, ncols, idx+1)
        plt.imshow(imdata[idx, :, :], cmap='gray')
        plt.xticks([], [])
        plt.yticks([], [])
        
def build_montage(images):
    '''
        Build a montage out of images
    '''
    nimg, H, W = images.shape
    
    nrows = int(np.ceil(np.sqrt(nimg)))
    ncols = int(np.ceil(nimg/nrows))
    
    montage_im = np.zeros((H*nrows, W*ncols), dtype=np.float32)
    
    cnt = 0
    for r in range(nrows):
        for c in range(ncols):
            h1 = r*H
            h2 = (r+1)*H
            w1 = c*W
            w2 = (c+1)*W

            if cnt == nimg:
                break

            montage_im[h1:h2, w1:w2] = images[cnt, ...]
            cnt += 1
    
    return montage_im
      
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def ims2rgb(im1, im2):
    '''
        Concatenate images into RGB
        
        Inputs:
            im1, im2: Two images to compare
    '''
    H, W = im1.shape
    
    imrgb = np.zeros((H, W, 3))
    imrgb[..., 0] = im1
    imrgb[..., 2] = im2

    return imrgb

def textfunc(im, txt):
    return cv2.putText(im, txt, (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (1, 1, 1),
                        2,
                        cv2.LINE_AA)
    
def get_img(imname, scaling):
    # Read image
    im = cv2.resize(plt.imread('../data/%s.png'%imname), None,
                    fx=scaling, fy=scaling)**2
    
    if im.ndim == 2:
        im = im[:, :, np.newaxis]
        im = im[:, :, [0, 0, 0]]
        im = np.copy(im, order='C')
    H, W, _ = im.shape
    
    return np.copy(im[..., 1], order='C').astype(np.float32)

def get_real_im(imname, camera):
    im = io.loadmat('../results/rawdata/%s/%s.mat'%(camera, imname))['imstack']
    #im = normalize(im['imstack'].astype(np.float32), True)
    minval = im.min()
    maxval = im.max()
    
    if camera == 'rgb':
        im = normalize(im[:, ::2, ::2], True)
        
        #nimg_full, H, W = im.shape
        #im_scaled = np.zeros((nimg_full, H//2, W//2), dtype=np.float32)
        #for idx in range(nimg_full):
        #    im_scaled[idx, ...] = cv2.resize(im[idx, ...], (W//2, H//2),
        #                                        interpolation=cv2.INTER_AREA)
        #im = im_scaled
    else:
        im = normalize(im, True).astype(np.float32)
        
    return im, minval, maxval

def boxify(im, topleft, boxsize, color=[1, 1, 1], width=2):
    '''
        Generate a box around a region.
    '''
    h, w = topleft
    dh, dw = boxsize
    
    im[h:h+dh+1, w:w+width, :] = color
    im[h:h+width, w:w+dh+width, :] = color
    im[h:h+dh+1, w+dw:w+dw+width, :] = color
    im[h+dh:h+dh+width, w:w+dh+width, :] = color

    return im

def resize(cube, scale):
    '''
        Resize a multi-channel image
        
        Inputs:
            cube: (H, W, nchan) image stack
            scale: Scaling 
    '''
    H, W, nchan = cube.shape
    
    im0_lr = cv2.resize(cube[..., 0], None, fx=scale, fy=scale)
    Hl, Wl = im0_lr.shape
    
    cube_lr = np.zeros((Hl, Wl, nchan), dtype=cube.dtype)
    
    for idx in range(nchan):
        cube_lr[..., idx] = cv2.resize(cube[..., idx], None,
                                       fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
    return cube_lr

def moduloclip(cube, mulsize):
    '''
        Clip a cube to have multiples of mulsize
        
        Inputs:
            cube: (H, W, T) sized cube
            mulsize: (h, w) tuple having smallest stride size
            
        Outputs:
            cube_clipped: Clipped cube with size equal to multiples of h, w
    '''
    if len(mulsize) == 2:
        H, W = cube.shape[:2]
        
        H1 = mulsize[0]*(H // mulsize[0])
        W1 = mulsize[1]*(W // mulsize[1])
        
        cube_clipped = cube[:H1, :W1]
    else:
        H, W, T = cube.shape
        H1 = mulsize[0]*(H // mulsize[0])
        W1 = mulsize[1]*(W // mulsize[1])
        T1 = mulsize[2]*(T // mulsize[2])
        
        cube_clipped = cube[:H1, :W1, :T1]
    
    return cube_clipped

def implay(cube, delay=20):
    '''
        Play hyperspectral image as a video
    '''
    if cube.dtype != np.uint8:
        cube = (255*cube/cube.max()).astype(np.uint8)
    
    T = cube.shape[-1]
    
    for idx in range(T):
        cv2.imshow('Video', cube[..., idx])
        cv2.waitKey(delay)
        
        
def get_matrix(nrows, ncols, rank, noise_type, signal_type,
               noise_snr=5, tau=1000):
    '''
        Get a matrix for simulations
        
        Inputs:
            nrows, ncols: Size of the matrix
            rank: Rank of the matrix
            noise_type: Type of the noise to add. Currently None, gaussian,
                and poisson
            signal_type: Type of the signal itself. Currently gaussian and
                piecewise constant
            noise_snr: Amount of noise to add in terms of 
    '''
    if signal_type == 'gaussian':
        U = np.random.randn(nrows, rank)
        V = np.random.randn(rank, ncols)
    elif signal_type == 'piecewise':
        nlocs = 10
        
        U = np.zeros((nrows, rank))
        V = np.zeros((rank, ncols))
        
        for idx in range(rank):
            u_locs = np.random.randint(0, nrows, nlocs)
            v_locs = np.random.randint(0, ncols, nlocs)
            
            U[u_locs, idx] = np.random.randn(nlocs)
            V[idx, v_locs] = np.random.randn(nlocs)
        
        U = np.cumsum(U, 0)
        V = np.cumsum(V, 1)
    else:
        raise AttributeError('Signal type not implemented')
    
    mat = normalize(U.dot(V), True)
    
    if noise_type == 'gaussian':
        mat_noisy = measure(mat, noise_snr, float('inf'))
    elif noise_type == 'poisson':
        mat_noisy = measure(mat, noise_snr, tau)
    elif noise_type == 'rician':
        noise1 = np.random.randn(nrows, ncols)*noise_snr
        noise2 = np.random.randn(nrows, ncols)*noise_snr

        mat_noisy = np.sqrt((mat + noise1)**2 + noise2**2)
    else:
        raise AttributeError('Noise type not implemented')
    
    return mat_noisy, mat

def get_pca(nrows, ndata, rank, noise_type, signal_type,
            noise_snr=5, tau=1000):
    '''
        Get PCA data
        
        Inputs:
            nrows: Number of rows in data
            ndata: Number of data points
            rank: Intrinsic dimension
            noise_type: Type of the noise to add. Currently None, gaussian,
                and poisson
            signal_type: Type of the signal itself. Currently gaussian and
                piecewise constant
            noise_snr: Amount of noise to add in terms of 
    '''
    # Generate normalized coefficients
    coefs = np.random.randn(rank, ndata)
    coefs_norm = np.sqrt((coefs*coefs).sum(0)).reshape(1, ndata)
    coefs = coefs/coefs_norm

    if signal_type == 'gaussian':
        basis = np.random.randn(nrows, rank)
    elif signal_type == 'piecewise':
        nlocs = 10
        
        basis = np.zeros((nrows, rank))
        
        for idx in range(rank):
            u_locs = np.random.randint(0, nrows, nlocs)            
            basis[u_locs, idx] = np.random.randn(nlocs)
        
        basis = np.cumsum(basis, 0)
    else:
        raise AttributeError('Signal type not implemented')
    
    # Compute orthogonal basis with QR decomposition
    basis, _ = np.linalg.qr(basis)
    mat = basis.dot(coefs)
    
    if noise_type == 'gaussian':
        mat_noisy = measure(mat, noise_snr, float('inf'))
    elif noise_type == 'poisson':
        mat_noisy = measure(mat, noise_snr, tau)
    elif noise_type == 'rician':
        noise1 = np.random.randn(nrows, ndata)*noise_snr
        noise2 = np.random.randn(nrows, ndata)*noise_snr

        mat_noisy = np.sqrt((mat + noise1)**2 + noise2**2)
    else:
        raise AttributeError('Noise type not implemented')
    
    return mat_noisy, mat, basis

def get_inp(tensize, const=10.0):
    '''
        Wrapper to get a variable on graph
    '''
    inp = torch.rand(tensize).cuda()/const
    inp = torch.autograd.Variable(inp, requires_grad=True).cuda()
    inp = torch.nn.Parameter(inp)
    
    return inp

def get_coords(H, W, T=None):
    '''
        Get 2D/3D coordinates
    '''
    if T is None:
        X, Y = np.meshgrid(np.linspace(-1, 1, W), np.linspace(-1, 1, H))
        coords = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
    else:
        X, Y, Z = np.meshgrid(np.linspace(-1, 1, W),
                              np.linspace(-1, 1, H),
                              np.linspace(-1, 1, T))
        coords = np.hstack((X.reshape(-1, 1),
                            Y.reshape(-1, 1),
                            Z.reshape(-1, 1)))
    
    return torch.tensor(coords.astype(np.float32)).cuda()

def get_3d_posencode_inp(H, W, T, n_inputs):
    '''
        Get positionally encoded inputs for inpainting tasks
        
        https://bmild.github.io/fourfeat/
    '''
    X, Y, Z = np.mgrid[:H, :W, :T]
    coords = np.stack((10*(X/H), 10*(Y/W), 10*(Z/T)), axis=3).reshape(-1, 3)
    
    freqs = np.random.rand(3, n_inputs)
    
    angles = coords.dot(freqs)
    
    sin_vals = np.sin(2*np.pi*angles)
    cos_vals = np.cos(2*np.pi*angles)
    
    #posencode_vals = np.hstack((sin_vals, cos_vals)).astype(np.float32)
    posencode_vals = sin_vals.astype(np.float32)
    
    inp = posencode_vals.reshape(H, W, T, n_inputs)
    inp = torch.autograd.Variable(torch.tensor(inp), requires_grad=True).cuda()
    inp = torch.nn.Parameter(inp.permute(3, 2, 0, 1)[None, ...])
    
    return inp

def get_2d_posencode_inp(H, W, n_inputs):
    '''
        Get positionally encoded inputs for inpainting tasks
        
        https://bmild.github.io/fourfeat/
    '''
    X, Y = np.mgrid[:H, :W]
    coords = np.hstack((10*(X/H).reshape(-1, 1), 10*(Y/W).reshape(-1, 1)))
    
    freqs = np.random.rand(2, n_inputs)
    
    angles = coords.dot(freqs)
    
    sin_vals = np.sin(2*np.pi*angles)
    cos_vals = np.cos(2*np.pi*angles)
    
    posencode_vals = np.hstack((sin_vals, cos_vals)).astype(np.float32)
    
    inp = posencode_vals.reshape(H, W, 2*n_inputs)
    inp = torch.autograd.Variable(torch.tensor(inp), requires_grad=True).cuda()
    inp = torch.nn.Parameter(inp.permute(2, 0, 1)[None, ...])
    
    return inp

def get_1d_posencode_inp(H, n_inputs):
    '''
        Get positionally encoded inputs for inpainting tasks
        
        https://bmild.github.io/fourfeat/
    '''
    X = np.arange(H).reshape(-1, 1)
    
    freqs = np.random.rand(1, n_inputs)
    
    angles = (10*X/H).dot(freqs)
    
    sin_vals = np.sin(2*np.pi*angles)
    cos_vals = np.cos(2*np.pi*angles)
    
    posencode_vals = np.hstack((sin_vals, cos_vals)).astype(np.float32)
    
    inp = posencode_vals.reshape(H, 2*n_inputs)
    inp = torch.autograd.Variable(torch.tensor(inp), requires_grad=True).cuda()
    inp = torch.nn.Parameter(inp.permute(1, 0)[None, ...])
    
    return inp  

def get_1d_coords_inp(H, n_inputs):
    '''
        Get smooth 2D coordinate input
    '''
    X, _ = np.mgrid[:n_inputs, :H]
    
    X = X.astype(np.float32)/(10*max(H, n_inputs))
    
    inp = torch.tensor(X).cuda()[None, ...]
    inp = torch.autograd.Variable(torch.tensor(inp), requires_grad=True).cuda()
    inp = torch.nn.Parameter(inp)
    
    return inp  

def lr_decompose(mat, rank=6):
    '''
        Low rank decomposition
    '''
    u, s, vt = linalg.svds(mat, k=rank)
    mat_lr = u.dot(np.diag(s)).dot(vt)
    
    return mat_lr

def get_scheduler(scheduler_type, optimizer, args):
    '''
        Get a scheduler 
        
        Inputs:
            scheduler_type: 'none', 'step', 'exponential', 'cosine'
            optimizer: One of torch.optim optimizers
            args: Namspace containing arguments relevant to each optimizer
            
        Outputs:
            scheduler: A torch learning rate scheduler
    '''
    if scheduler_type == 'none':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.epochs)
    elif scheduler_type == 'step':
        # Compute gamma 
        gamma = pow(10, -1/(args.epochs/args.step_size))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=args.step_size,
                                                    gamma=gamma)
    elif scheduler_type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=args.gamma)
    elif scheduler_type == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                first_cycle_steps=200,
                                                cycle_mult=1.0,
                                                max_lr=args.max_lr,
                                                min_lr=args.min_lr,
                                                warmup_steps=50,
                                                gamma=args.gamma)
        
    return scheduler

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_h5(file_path):
    hdf5 = h5py.File(file_path, 'r')
    ct_data = np.asarray(hdf5['ct'])
    x_ray1 = np.asarray(hdf5['xray1'])
    x_ray1 = np.expand_dims(x_ray1, 0)  # (1, 256, 256)
    hdf5.close()
    return ct_data, x_ray1