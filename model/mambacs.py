import torch
from torch import nn
from model.mamba_r import Mamba_R
from model.attention import BSAM, SaELayer
from model.sampling import CS_Sampling

class MambaCS(nn.Module):
    def __init__(self, sensing_rate, im_size):
        super(MambaCS, self).__init__()

        self.sensing_rate = sensing_rate
        self.base = 64
        self.blocksize = 32

        if sensing_rate == 0.5 :
            self.cs_sampling = CS_Sampling(n_channels=3, cs_ratio=sensing_rate, blocksize=16, im_size=im_size) 

            self.mamba_r = Mamba_R(upscale=1, in_chans=3, img_size=im_size, img_range=1., d_state=16, 
                                   depths=[6, 6, 6, 6], embed_dim=384, mlp_ratio=1.2)
        if sensing_rate == 0.25 :
            self.cs_sampling = CS_Sampling(n_channels=3, cs_ratio=sensing_rate, blocksize=16, im_size=im_size) 

            self.mamba_r = Mamba_R(upscale=1, in_chans=3, img_size=im_size, img_range=1., d_state=16, 
                                   depths=[8, 8, 8, 8], embed_dim=192, mlp_ratio=1.2)
        if sensing_rate == 0.1 :
            self.cs_sampling = CS_Sampling(n_channels=3, cs_ratio=sensing_rate, blocksize=16, im_size=im_size) 

            self.mamba_r = Mamba_R(upscale=1, in_chans=3, img_size=im_size, img_range=1., d_state=16, 
                                   depths=[8, 8, 8, 8], embed_dim=75, mlp_ratio=1.2)
        if sensing_rate == 0.04 :
            self.cs_sampling = CS_Sampling(n_channels=3, cs_ratio=sensing_rate, blocksize=16, im_size=im_size) 

            self.mamba_r = Mamba_R(upscale=1, in_chans=3, img_size=im_size, img_range=1., d_state=16, 
                                   depths=[8, 8, 8, 8], embed_dim=30, mlp_ratio=1.2)
        if sensing_rate == 0.01 :
            self.cs_sampling = CS_Sampling(n_channels=3, cs_ratio=sensing_rate, blocksize=32, im_size=im_size) 

            self.mamba_r = Mamba_R(upscale=1, in_chans=3, img_size=im_size, img_range=1., d_state=16, 
                                   depths=[8, 8, 8, 8], embed_dim=30, mlp_ratio=1.2)

    def forward(self, x):
        initial, xsample = self.cs_sampling(x)
        out = self.mamba_r(initial, xsample)
        return out, initial
