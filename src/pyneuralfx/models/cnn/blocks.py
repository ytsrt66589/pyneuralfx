import torch
import torch.nn as nn

from ..utils import * 
from .basic_block import *



class TCNBlock(nn.Module):
    def __init__(
        self,
        n_inp: int, 
        n_output: int, 
        n_cond: int,
        kernel_size: int,
        dilation: int, 
        causal: bool,
        filmed: bool, 
        hypered: bool,
        bias: bool,
    ):
        super().__init__()
        self.causal = causal 
        self.filmed = filmed
        self.hypered = hypered

        if hypered:
            self.conv1 = HyperConv(
                n_inp,
                n_output,
                kernel_size,
                dilation,
                n_cond,
                bias=bias,
            )
        else:
            self.conv1 = nn.Conv1d(n_inp, 
                                    n_output, 
                                    kernel_size = kernel_size, 
                                    padding = 0, # zero padding
                                    dilation = dilation,
                                    bias = bias) # better initialization 

        if filmed and n_cond > 0:
            self.affine = FiLM(n_cond, n_output)
        else:
            self.affine = nn.BatchNorm1d(n_output)
        
        self.activation_function = nn.LeakyReLU(0.2) 
        self.res = nn.Conv1d(n_inp, 
                            n_output, 
                            kernel_size=1,
                            bias=False)
        
    def forward(self, x, c, *args):
        x_in = x

        if self.hypered:
            x = self.conv1(x, c)
        else:
            x = self.conv1(x)

        if c is not None and self.filmed: 
            x = self.affine(x, c)
        else:
            x = self.affine(x)

        x = self.activation_function(x)
        x_res = self.res(x_in)

        if self.causal: 
            x = x + causal_crop(x_res, x.shape[-1])
        else:
            x = x + center_crop(x_res, x.shape[-1])
        return x


