import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from torchdiffeq import odeint_adjoint as odeint

import os 

tol = 1e-2

class ODEBlock_image(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock_image, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol)
        return  out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
        
        
class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut
    
    
class ODEfunc_1(nn.Module):

    def __init__(self,dim):
        super(ODEfunc_1, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(self.dim, self.dim, 3, 1, 1)
        self.norm1 = norm(self.dim)
        self.conv2 = nn.Conv2d(self.dim, self.dim, 3, 1, 1)
        self.norm2 = norm(self.dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, t, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)
        return out
    
class CNN_ode(nn.Module):
    def __init__(self, downsampling_method='res'):
        super(CNN_ode, self).__init__()
        self.downsampling_method = downsampling_method
        if  self.downsampling_method == 'conv':
            downsampling_layers = [
                nn.Conv2d(3, 64, 3, 1),
                norm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 4, 2, 1),
                norm(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 4, 2, 2),
            ]
        elif self.downsampling_method == 'res':
            downsampling_layers = [
                nn.Conv2d(3, 64, 3, 1),
                ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
                ResBlock(64, 128, stride=2, downsample=conv1x1(64, 128, 2)),
            ]
            
        feature_layers = [ODEBlock_image(ODEfunc_1(128))]
        fc_layers = [norm(128), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(128, 512)]
        self.CNN_model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers)#.to(device)
        
    def forward(self, x_meas_src):
        feature_out = self.CNN_model(x_meas_src)
        
        return feature_out