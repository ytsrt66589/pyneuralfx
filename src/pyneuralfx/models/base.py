import torch 
import torch.nn as nn 

from abc import ABC, abstractmethod


class NN_Base(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def compute_receptive_field(self):
        raise NotImplementedError
    
    def compute_num_of_params(self):
        return (sum(p.numel() for p in self.parameters()), sum(p.numel() for p in self.parameters() if p.requires_grad))


class CNN_Base(NN_Base):
    def __init__(self, kernel_size, sample_rate, dilation_growth, n_blocks):
        super().__init__()
        self.kernel_size = kernel_size 
        self.sample_rate = sample_rate
        self.dilation_growth = dilation_growth
        self.n_blocks = n_blocks

        
    def compute_receptive_field(self): # in samples 
        rf = self.kernel_size
        for n in range(1, self.n_blocks):
            dilation = self.dilation_growth ** n
            rf = rf + ((self.kernel_size-1) * dilation)
        
        return rf, (rf/self.sample_rate) * 1000 # samples, ms

class RNN_Base(NN_Base):
    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate

    def compute_receptive_field(self): # in samples 
        return 1, (1/self.sample_rate) * 1000 # samples, ms