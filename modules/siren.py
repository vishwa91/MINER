#!/usr/bin/env python

import os
import sys
from torch.functional import align_tensors
from torch.nn.modules.linear import Linear
import tqdm
import pdb
import math

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
#from pytorch_wavelets import DWTForward, DWTInverse

import skimage
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
    
class TanhLayer(nn.Module):
    '''
        Drop in repalcement for SineLayer but with Tanh nonlinearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
    def forward(self, input):
        return torch.tanh(self.linear(input))

class ReLULayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with ReLU non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return nn.functional.relu(self.linear(input))
    
class SincLayer(nn.Module):
    '''
        Instead of a sinusoid, utilize a sync nonlinearity
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.eps = 1e-3
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        denom = self.omega_0*self.linear(input)
        numer = torch.cos(denom)
        
        return numer/(1 + abs(denom).pow(2) )
    
class SineLayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega_0.
    
        If is_first=True, omega_0 is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega_0 in the first layer - this is a hyperparameter.
    
        If is_first=False, then the weights will be divided by omega_0 so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class PosEncoding(nn.Module):
    '''Module to add positional encoding as in NeRF [Mildenhall et al. 2020].'''
    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(min(sidelength[0], sidelength[1]))
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)
        else:
            self.num_frequencies = 4

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2 ** i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2 ** i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, 
                 out_features, nonlinearity='sine', outermost_linear=False, first_omega_0=30, hidden_omega_0=30., pos_encode=False, 
                 sidelength=512, fn_samples=None, use_nyquist=True):
        super().__init__()
        self.pos_encode = pos_encode
        
        if nonlinearity == 'sine':
            self.nonlin = SineLayer
        elif nonlinearity == 'tanh':
            self.nonlin = TanhLayer
        elif nonlinearity == 'sinc':
            self.nonlin = SincLayer
        else:
            self.nonlin = ReLULayer
            
        if pos_encode:
            self.positional_encoding = PosEncoding(in_features=in_features,
                                                   sidelength=sidelength,
                                                   fn_samples=fn_samples,
                                                   use_nyquist=use_nyquist)
            in_features = self.positional_encoding.out_dim
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features,
                                        is_first=False,
                                        omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, 
                                     out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        # allows to take derivative w.r.t. input
        #coords = coords.clone().detach().requires_grad_(True) 
        if self.pos_encode:
            coords = self.positional_encoding(coords)
            
        output = self.net(coords)
        #return output, coords    
        return output
    
class LinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
    '''
    def __init__(self, input_size, output_size, channel_size):
        super(LinearWithChannel, self).__init__()
        
        #initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(channel_size,
                                                     input_size,
                                                     output_size))
        self.bias = torch.nn.Parameter(torch.zeros(channel_size,
                                                   1,
                                                   output_size))
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias)
        
    def reset_parameters(self, weights, bias):
        
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x):
        mul_output = torch.bmm(x, self.weight)
        return mul_output + self.bias
    
class AdaptiveLinearWithChannel(nn.Module):
    '''
        Implementation from https://github.com/pytorch/pytorch/issues/36591
        
        Evaluate only selective channels
    '''
    def __init__(self, input_size, output_size, channel_size):
        super(AdaptiveLinearWithChannel, self).__init__()
        
        #initialize weights
        self.weight = torch.nn.Parameter(torch.zeros(channel_size,
                                                     input_size,
                                                     output_size))
        self.bias = torch.nn.Parameter(torch.zeros(channel_size,
                                                   1,
                                                   output_size))
        
        #change weights to kaiming
        self.reset_parameters(self.weight, self.bias)
        
    def reset_parameters(self, weights, bias):
        
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)
    
    def forward(self, x, indices):
        #mul_output = torch.bmm(x, self.weight[indices, ...])
        return torch.bmm(x, self.weight[indices, ...]) + self.bias[indices, ...]
    
class MultiSineLayer(nn.Module):
    '''
        Implements sinusoidal activations with multiple channel input
    '''
    
    def __init__(self, in_features, out_features, n_channels, is_first=False, 
                 omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        
        
        self.linear = LinearWithChannel(in_features, out_features, n_channels)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
class AdaptiveMultiSineLayer(nn.Module):
    '''
        Implements sinusoidal activations with multiple channel input
    '''
    
    def __init__(self, in_features, out_features, n_channels, is_first=False, 
                 omega_0=30, const=1.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.const = const
        
        self.in_features = in_features
        
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = self.const/self.in_features
                self.linear.weight.uniform_(-bound, bound)      
                self.linear.bias.uniform_(-bound, bound)
            else:
                bound = np.sqrt(self.const*6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                self.linear.bias.uniform_(-bound, bound)
        
    def forward(self, input, indices):
        return torch.sin(self.omega_0 * self.linear(input, indices))    
class AdaptiveMultiReLULayer(nn.Module):
    '''
        Implements ReLU activations with multiple channel input.
        
        The parameters is_first, and omega_0 are not relevant.
    '''
    
    def __init__(self, in_features, out_features, n_channels, is_first=False, 
                 omega_0=30):
        super().__init__()        
        self.in_features = in_features
        self.linear = AdaptiveLinearWithChannel(in_features,
                                                out_features,
                                                n_channels)
        self.relu = torch.nn.LeakyReLU()
        
    def forward(self, input, indices):
        return self.relu(self.linear(input, indices))    
class MultiSiren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, 
                 out_features, n_channels, outermost_linear=False, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.nonlin = MultiSineLayer
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, n_channels,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        n_channels, is_first=False, 
                                        omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = LinearWithChannel(hidden_features,
                                             out_features,
                                             n_channels)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                        n_channels, is_first=False, 
                                        omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):            
        output = self.net(coords) 
        return output
    
class MultiSequential(nn.Sequential):
    '''
        https://github.com/pytorch/pytorch/issues/19808#
    '''
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
    
class AdaptiveMultiSiren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, 
                 out_features, n_channels, outermost_linear=False, first_omega_0=30, hidden_omega_0=30., nonlin='sine',
                 const=1.0):
        super().__init__()
        
        if nonlin == 'sine':
            self.nonlin = AdaptiveMultiSineLayer
        elif nonlin == 'relu':
            self.nonlin = AdaptiveMultiReLULayer
            
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, n_channels,
                                  is_first=True, omega_0=first_omega_0,
                                  const=const))

        for i in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features,
                                        hidden_features, 
                                        n_channels, is_first=False, 
                                        omega_0=hidden_omega_0,
                                        const=const))

        if outermost_linear:
            feat = hidden_features
            final_linear = AdaptiveLinearWithChannel(feat, 
                                                     out_features,
                                                     n_channels)
            
            if nonlin == 'sine':
                with torch.no_grad():
                    bound = np.sqrt(const / hidden_features) / hidden_omega_0
                    final_linear.weight.uniform_(-bound, bound)
                    final_linear.bias.uniform_(-bound, bound)
                    
            self.net.append(final_linear)
        else:
            self.net.append(self.nonlin(hidden_features, out_features, 
                                        n_channels, is_first=False, 
                                        omega_0=hidden_omega_0,
                                        const=const))
        
        self.net = nn.ModuleList(self.net)
    
    def forward(self, inp, indices):            
        output = inp[indices, ...]

        for mod in self.net:
            output = mod(output, indices)
        return output
    