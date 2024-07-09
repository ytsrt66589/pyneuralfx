import torch
import torch.nn as nn

from ..utils import * 
from .basic_block import *
from ..filters.analog_filters import FreqRespSampConv1d

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

class AnalogTCNBlock(nn.Module):
    def __init__(
        self,
        n_inp: int, 
        n_output: int, 
        n_cond: int,
        kernel_size: int,
        dilation: int, 
        causal: bool,
        bias: bool,
        n_samples: int = 640
    ):
        super().__init__()
        self.out_ch = n_output 
        self.causal = causal
        
        self.conv1 = FreqRespSampConv1d(
            in_channels=n_inp,
            out_channels=n_output,
            n_samples=n_samples,
            dilation=dilation,
        )

        self.affine = nn.BatchNorm1d(n_output)

        self.activation_function = nn.LeakyReLU(0.2) 
        self.res = nn.Conv1d(n_inp, 
                            n_output, 
                            kernel_size=1,
                            bias=False)
        
    def forward(self, x, c, *args):
        x_in = x

        x = self.conv1(x)

        x = self.affine(x)

        x = self.activation_function(x)
        x_res = self.res(x_in)

        if self.causal: 
            x = x + causal_crop(x_res, x.shape[-1])
        else:
            x = x + center_crop(x_res, x.shape[-1])
        return x

    def prepare(self, sample_rate, kernel_size, stride=1):
        self.conv1.prepare(sample_rate, kernel_size, stride)

class GCNBlock(nn.Module):
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
        self.out_ch = n_output 

        if hypered:
            self.conv1 = HyperConv(
                n_inp,
                n_output * 2,
                kernel_size,
                dilation,
                n_cond,
                bias=bias,
            )
        else:
            self.conv1 = nn.Conv1d(n_inp, 
                                    n_output * 2, 
                                    kernel_size = kernel_size, 
                                    padding = 0, # zero padding
                                    dilation = dilation,
                                    bias = bias) # better initialization 

        if filmed and n_cond > 0:
            self.affine = FiLM(n_cond, n_output)
        
        
        self.res = nn.Conv1d(n_inp, 
                            n_output, 
                            kernel_size=1,
                            bias=False)
        
        self.mix = torch.nn.Conv1d(n_output,
                                   n_output,
                                   kernel_size = 1,
                                   stride = 1,
                                   padding = 0,
                                   bias = bias)
        
    def forward(self, x, c, *args):
        x_in = x

        if self.hypered:
            x = self.conv1(x, c)
        else:
            x = self.conv1(x)

        # gated mechanism 
        x = torch.tanh(x[:, :self.out_ch, :]) * torch.sigmoid(x[:, self.out_ch:, :])

        if c is not None and self.filmed: 
            x = self.affine(x, c)
        
        z = x 

        x = self.mix(x) 
        x_res = self.res(x_in)

        if self.causal: 
            x = x + causal_crop(x_res, x.shape[-1])
        else:
            x = x + center_crop(x_res, x.shape[-1])
        
        return x, z 
    
class AnalogGCNBlock(nn.Module):
    def __init__(
        self,
        n_inp: int, 
        n_output: int, 
        n_cond: int,
        kernel_size: int,
        dilation: int, 
        causal: bool,
        bias: bool,
        n_samples: int = 640
    ):
        super().__init__()
        self.out_ch = n_output 
        self.causal = causal

        self.conv1 = FreqRespSampConv1d(
            in_channels=n_inp,
            out_channels=n_output * 2,
            n_samples=n_samples,
            dilation=dilation,
        )

        self.res = nn.Conv1d(n_inp, 
                            n_output, 
                            kernel_size=1,
                            bias=False)
        
        self.mix = torch.nn.Conv1d(n_output,
                                   n_output,
                                   kernel_size = 1,
                                   stride = 1,
                                   padding = 0,
                                   bias = bias)
        
    def forward(self, x, c, *args):
        x_in = x

        x = self.conv1(x)

        # gated mechanism 
        x = torch.tanh(x[:, :self.out_ch, :]) * torch.sigmoid(x[:, self.out_ch:, :])

        z = x 

        x = self.mix(x) 
        x_res = self.res(x_in)

        if self.causal: 
            x = x + causal_crop(x_res, x.shape[-1])
        else:
            x = x + center_crop(x_res, x.shape[-1])
        
        return x, z 

    def prepare(self, sample_rate, kernel_size, stride=1):
        self.conv1.prepare(sample_rate, kernel_size, stride)

