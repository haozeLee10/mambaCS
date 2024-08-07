import torch
import torch.nn as nn
from torch.nn import init
import cv2
import numpy as np
import random
import torch.nn.functional as F


class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        #         print(input_.shape)
        grad_input = grad_output.clone()
        grad_input[input_ < -1] = 0
        grad_input[input_ > 1] = 0
        return grad_input


MyBinarize = MySign.apply


class CS_Sampling(torch.nn.Module):
    def __init__(self, n_channels=3, cs_ratio=0.5, blocksize=16, im_size=64):
        super(CS_Sampling, self).__init__()
        print('bcs')

        n_output = int(blocksize ** 2)
        n_input = int(cs_ratio * n_output)

        self.PhiR = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))
        self.PhiG = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))
        self.PhiB = nn.Parameter(init.xavier_normal_(torch.Tensor(n_input, n_output)))

        self.n_channels = n_channels
        self.n_input = n_input
        self.n_output = n_output
        self.blocksize = blocksize

        self.im_size = im_size

    def forward(self, x):
        Phi_R = self.PhiR
        Phi_G = self.PhiG
        Phi_B = self.PhiB

        PhiWeight_R = Phi_R.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_G = Phi_G.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)
        PhiWeight_B = Phi_B.contiguous().view(int(self.n_input), 1, self.blocksize, self.blocksize)

        Phix_R = F.conv2d(x[:, 0:1, :, :], PhiWeight_R, padding=0, stride=self.blocksize, bias=None)  # Get measurements
        Phix_G = F.conv2d(x[:, 1:2, :, :], PhiWeight_G, padding=0, stride=self.blocksize, bias=None)  # Get measurements
        Phix_B = F.conv2d(x[:, 2:3, :, :], PhiWeight_B, padding=0, stride=self.blocksize, bias=None)  # Get measurements

        # Initialization-subnet
        PhiTWeight_R = Phi_R.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_R = F.conv2d(Phix_R, PhiTWeight_R, padding=0, bias=None)
        PhiTb_R = torch.nn.PixelShuffle(self.blocksize)(PhiTb_R)
        x_R = PhiTb_R  # Conduct initialization

        PhiTWeight_G = Phi_G.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_G = F.conv2d(Phix_G, PhiTWeight_G, padding=0, bias=None)
        PhiTb_G = torch.nn.PixelShuffle(self.blocksize)(PhiTb_G)
        x_G = PhiTb_G

        PhiTWeight_B = Phi_B.t().contiguous().view(self.n_output, self.n_input, 1, 1)
        PhiTb_B = F.conv2d(Phix_B, PhiTWeight_B, padding=0, bias=None)
        PhiTb_B = torch.nn.PixelShuffle(self.blocksize)(PhiTb_B)
        x_B = PhiTb_B

        x = torch.cat([x_R, x_G, x_B], dim=1)
        x = F.interpolate(x, size=(self.im_size, self.im_size), mode='bilinear')

        xsample = torch.cat([Phix_R, Phix_G, Phix_B], dim=1)
        xsample = F.interpolate(xsample, size=(self.im_size, self.im_size), mode='bilinear')
        
        return x, xsample

