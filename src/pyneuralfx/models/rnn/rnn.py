import torch 
import torch.nn as nn 
from .base import * 

class SnapShotVanillaRNN(SnapShot_Base):
    def __init__(
        self,
        inp_channel: int,
        out_channel: int,
        rnn_size: int,
        sample_rate: int, 
        layer_norm: bool = False,
        rnn_bias: bool = True, 
    ):
        super().__init__(
            inp_channel = inp_channel, 
            out_channel = out_channel, 
            rnn_size = rnn_size, 
            cell_type = 'vanilla_rnn', 
            sample_rate = sample_rate, 
            layer_norm = layer_norm, 
            rnn_bias = rnn_bias
        )



class ConcatVanillaRNN(Concat_Base):
    def __init__(
        self,
        inp_channel: int,
        out_channel: int,
        rnn_size: int,
        sample_rate: int, 
        num_conds: int = 0,
        layer_norm: bool = False,
        rnn_bias: bool = True, 
    ):
        super().__init__(
            inp_channel = inp_channel, 
            out_channel = out_channel, 
            rnn_size = rnn_size, 
            cell_type = 'vanilla_rnn', 
            sample_rate = sample_rate, 
            num_conds = num_conds,
            layer_norm = layer_norm, 
            rnn_bias = rnn_bias
        )

class FiLMVanillaRNN(FiLM_Base):
    def __init__(
        self,
        inp_channel: int,
        out_channel: int,
        rnn_size: int,
        sample_rate: int, 
        n_mlp_blocks: int = 3,
        mlp_size: int = 8,
        num_conds: int = 0,
        layer_norm: bool = False,
        rnn_bias: bool = True, 
    ):
        super().__init__(
            inp_channel = inp_channel, 
            out_channel = out_channel, 
            rnn_size = rnn_size, 
            cell_type = 'vanilla_rnn', 
            sample_rate = sample_rate, 
            n_mlp_blocks = n_mlp_blocks, 
            mlp_size = mlp_size,
            num_conds = num_conds,
            layer_norm = layer_norm, 
            rnn_bias = rnn_bias,
            filmed = True
        )

class StaticHyperVanillaRNN(StaticHyper_Base):
    def __init__(
        self,
        inp_channel: int,
        out_channel: int,
        rnn_size: int,
        sample_rate: int, 
        n_mlp_blocks: int = 3,
        mlp_size: int = 8,
        num_conds: int = 0,
        layer_norm: bool = False,
        rnn_bias: bool = True, 
    ):
        super().__init__(
            inp_channel = inp_channel, 
            out_channel = out_channel, 
            rnn_size = rnn_size, 
            cell_type = 'vanilla_rnn', 
            sample_rate = sample_rate, 
            n_mlp_blocks = n_mlp_blocks, 
            mlp_size = mlp_size,
            num_conds = num_conds,
            layer_norm = layer_norm, 
            rnn_bias = rnn_bias
        )

