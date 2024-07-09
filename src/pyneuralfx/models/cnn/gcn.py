import torch 
import torch.nn as nn 


from ..utils import * 
from .blocks import GCNBlock, AnalogGCNBlock


class SnapShotGCN(nn.Module):
    def __init__(
        self,
        n_inp: int, 
        n_output: int, 
        n_cond: int,
        sample_rate: int,
        n_blocks: int = 9, 
        kernel_size: int = 3, 
        dilation_growth: int = 2, 
        n_channels: int = 16, 
        causal: bool = True,  
    ):
        super().__init__()

        # args 
        self.n_inp = n_inp
        self.n_output = n_output
        self.n_cond = 0
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.n_channels = n_channels
        self.causal = causal 
        
        self.sample_rate = sample_rate # can used to prevent samplerate mismatch

        # modules 
        self.blocks = torch.nn.ModuleList()
        for n in range(self.n_blocks):
            in_ch = out_ch if n > 0 else self.n_inp
            out_ch = self.n_channels
            dilation = dilation_growth ** n 
            bias = False if n == 0 else True
            self.blocks.append(
                GCNBlock(
                    in_ch, 
                    out_ch, 
                    n_cond = 0,
                    kernel_size= kernel_size, 
                    dilation=dilation,
                    causal=causal,
                    filmed= False,
                    hypered = False,
                    bias = bias))


        # output mixing layer
        self.mixing_output = torch.nn.Conv1d(
            in_channels=self.n_channels * n_blocks,
            out_channels=self.n_output,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x, c, *args):
        
        skips = []
        # iterate over blocks passing conditioning
        for _, block in enumerate(self.blocks):
            x, zn = block(x, None)
            skips.append(zn)
        
        z = torch.cat([s[:, :, -x.size(2):] for s in skips], dim=1)

        out = self.mixing_output(z)
        
        return out

    def compute_receptive_field(self): # in samples 
        rf = self.kernel_size
        for n in range(1, self.n_blocks):
            dilation = self.dilation_growth ** n
            rf = rf + ((self.kernel_size-1) * dilation)
        return rf, rf/self.sample_rate * 1000 # samples, ms
    

    def compute_num_of_params(self):
        return (sum(p.numel() for p in self.parameters()), sum(p.numel() for p in self.parameters() if p.requires_grad))

class SnapShotAnalogGCN(nn.Module):
    def __init__(
        self,
        n_inp: int, 
        n_output: int, 
        n_cond: int,
        sample_rate: int,
        n_blocks: int = 9, 
        kernel_size: int = 3, 
        dilation_growth: int = 2, 
        n_channels: int = 16, 
        causal: bool = True,  
        n_samples: int = 640
    ):
        super().__init__()

        # args 
        self.n_inp = n_inp
        self.n_output = n_output
        self.n_cond = 0
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.n_channels = n_channels
        self.causal = causal 
        
        self.sample_rate = sample_rate # can used to prevent samplerate mismatch

        # modules 
        self.blocks = torch.nn.ModuleList()
        for n in range(self.n_blocks):
            in_ch = out_ch if n > 0 else self.n_inp
            out_ch = self.n_channels
            dilation = dilation_growth ** n 
            bias = False if n == 0 else True
            self.blocks.append(
                AnalogGCNBlock(
                    in_ch, 
                    out_ch, 
                    n_cond = 0,
                    kernel_size= kernel_size, 
                    dilation=dilation,
                    causal=causal,
                    bias = bias,
                    n_samples = n_samples))


        # output mixing layer
        self.mixing_output = torch.nn.Conv1d(
            in_channels=self.n_channels * n_blocks,
            out_channels=self.n_output,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.prepare(self.sample_rate, self.kernel_size, 1)
    def forward(self, x, c, *args):
        
        skips = []
        # iterate over blocks passing conditioning
        for _, block in enumerate(self.blocks):
            x, zn = block(x, None)
            skips.append(zn)
        
        z = torch.cat([s[:, :, -x.size(2):] for s in skips], dim=1)

        out = self.mixing_output(z)
        
        return out

    def compute_receptive_field(self): # in samples 
        rf = self.kernel_size
        for n in range(1, self.n_blocks):
            dilation = self.dilation_growth ** n
            rf = rf + ((self.kernel_size-1) * dilation)
        return rf, rf/self.sample_rate * 1000 # samples, ms
    

    def compute_num_of_params(self):
        return (sum(p.numel() for p in self.parameters()), sum(p.numel() for p in self.parameters() if p.requires_grad))

    def prepare(self, sample_rate, kernel_size, stride):
        for b in self.blocks:
            b.prepare(sample_rate, kernel_size, stride)

class ConcatGCN(nn.Module):
    def __init__(
        self,
        n_inp: int, 
        n_output: int, 
        n_cond: int,
        sample_rate: int,
        n_blocks: int = 9, 
        kernel_size: int = 3, 
        dilation_growth: int = 2, 
        n_channels: int = 16, 
        causal: bool = True,  
    ):
        super().__init__()

        # args 
        self.n_inp = n_inp
        self.n_output = n_output
        self.n_cond = n_cond
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.n_channels = n_channels
        self.causal = causal 
        
        self.sample_rate = sample_rate # can used to prevent samplerate mismatch

        # modules 
        self.blocks = torch.nn.ModuleList()
        for n in range(self.n_blocks):
            in_ch = out_ch if n > 0 else (self.n_inp + self.n_cond)
            out_ch = self.n_channels
            dilation = dilation_growth ** n 
            bias = False if n == 0 else True
            self.blocks.append(
                GCNBlock(
                    in_ch, 
                    out_ch, 
                    n_cond = 0,
                    kernel_size= kernel_size, 
                    dilation=dilation,
                    causal=causal,
                    filmed= False,
                    hypered = False,
                    bias = bias))


        # output mixing layer
        self.mixing_output = torch.nn.Conv1d(
            in_channels=self.n_channels * n_blocks,
            out_channels=self.n_output,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x, c, *args):

        if c is not None and len(c.shape) == 2:
            c = c.unsqueeze(-1)
            c = c.expand(-1, -1, x.shape[-1])

        # cat x and cond
        if self.n_cond > 0:
            x = torch.cat((x, c), dim=1)
        
        skips = []
        # iterate over blocks passing conditioning
        for _, block in enumerate(self.blocks):
            x, zn = block(x, None)
            skips.append(zn)
        
        z = torch.cat([s[:, :, -x.size(2):] for s in skips], dim=1)

        out = self.mixing_output(z)
        
        return out

    def compute_receptive_field(self): # in samples 
        rf = self.kernel_size
        for n in range(1, self.n_blocks):
            dilation = self.dilation_growth ** n
            rf = rf + ((self.kernel_size-1) * dilation)
        return rf, rf/self.sample_rate * 1000 # samples, ms
    

    def compute_num_of_params(self):
        return (sum(p.numel() for p in self.parameters()), sum(p.numel() for p in self.parameters() if p.requires_grad))
    
class FiLMGCN(nn.Module):
    def __init__(
        self,
        n_inp: int, 
        n_output: int, 
        n_cond: int,
        sample_rate: int,
        n_blocks: int = 9, 
        kernel_size: int = 3, 
        dilation_growth: int = 2, 
        n_channels: int = 16, 
        causal: bool = True,  
        pre_film_size: int = 16,
        pre_film_blocks: int = 3,
    ):
        super().__init__()

        # args 
        self.n_inp = n_inp
        self.n_output = n_output
        self.n_cond = n_cond
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.n_channels = n_channels
        self.causal = causal 
        
        self.sample_rate = sample_rate # can used to prevent samplerate mismatch

        # modules 
        self.pre_film = None
        if self.n_cond > 0:
            pre_film = torch.nn.ModuleList()
            for n in range(pre_film_blocks-1):
                inp = pre_film_size if n > 0 else n_cond
                out = pre_film_size
                pre_film.append(nn.Sequential(
                    nn.Linear(inp, out),
                    nn.LeakyReLU(0.2)
                ))
            pre_film.append(nn.Linear(pre_film_size, pre_film_size))
            self.pre_film = nn.Sequential(*pre_film)

        self.blocks = torch.nn.ModuleList()
        for n in range(self.n_blocks):
            in_ch = out_ch if n > 0 else self.n_inp
            out_ch = self.n_channels
            dilation = dilation_growth ** n 
            bias = False if n == 0 else True
            self.blocks.append(
                GCNBlock(
                    in_ch, 
                    out_ch, 
                    n_cond = pre_film_size,
                    kernel_size = kernel_size, 
                    dilation = dilation,
                    causal = causal,
                    filmed = True,
                    hypered = False,
                    bias = bias))

        # output mixing layer
        self.mixing_output = torch.nn.Conv1d(
            in_channels=self.n_channels * n_blocks,
            out_channels=self.n_output,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x, c, *args):
        
        if self.pre_film is not None and self.n_cond > 0:
            c = self.pre_film(c)
        
        skips = []
        # iterate over blocks passing conditioning
        for _, block in enumerate(self.blocks):
            x, zn = block(x, c)
            skips.append(zn)
        
        z = torch.cat([s[:, :, -x.size(2):] for s in skips], dim=1)

        out = self.mixing_output(z)
        
        return out

    def compute_receptive_field(self): # in samples 
        rf = self.kernel_size
        for n in range(1, self.n_blocks):
            dilation = self.dilation_growth ** n
            rf = rf + ((self.kernel_size-1) * dilation)
        return rf, rf/self.sample_rate * 1000 # samples, ms
    

    def compute_num_of_params(self):
        return (sum(p.numel() for p in self.parameters()), sum(p.numel() for p in self.parameters() if p.requires_grad))

class HyperGCN(nn.Module):
    def __init__(
        self,
        n_inp: int, 
        n_output: int, 
        n_cond: int,
        sample_rate: int,
        n_blocks: int = 9, 
        kernel_size: int = 3, 
        dilation_growth: int = 2, 
        n_channels: int = 16, 
        causal: bool = True, 
    ):
        super().__init__()

        # args 
        self.n_inp = n_inp
        self.n_output = n_output
        self.n_cond = n_cond
        self.n_blocks = n_blocks
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.n_channels = n_channels
        self.causal = causal 
        
        self.sample_rate = sample_rate # can used to prevent samplerate mismatch

        # modules 
        self.blocks = torch.nn.ModuleList()
        for n in range(self.n_blocks):
            in_ch = out_ch if n > 0 else self.n_inp
            out_ch = self.n_channels
            dilation = dilation_growth ** n 
            bias = False if n == 0 else True
            self.blocks.append(
                GCNBlock(
                    in_ch, 
                    out_ch, 
                    n_cond = n_cond,
                    kernel_size = kernel_size, 
                    dilation = dilation,
                    causal = causal,
                    filmed = False,
                    hypered = True,
                    bias = bias))

        # output mixing layer
        self.mixing_output = torch.nn.Conv1d(
            in_channels=self.n_channels * n_blocks,
            out_channels=self.n_output,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x, c, *args):
        
        skips = []
        # iterate over blocks passing conditioning
        for _, block in enumerate(self.blocks):
            x, zn = block(x, c)
            skips.append(zn)
        
        z = torch.cat([s[:, :, -x.size(2):] for s in skips], dim=1)

        out = self.mixing_output(z)
        
        return out

    def compute_receptive_field(self): # in samples 
        rf = self.kernel_size
        for n in range(1, self.n_blocks):
            dilation = self.dilation_growth ** n
            rf = rf + ((self.kernel_size-1) * dilation)
        return rf, rf/self.sample_rate * 1000 # samples, ms
    

    def compute_num_of_params(self):
        return (sum(p.numel() for p in self.parameters()), sum(p.numel() for p in self.parameters() if p.requires_grad))


