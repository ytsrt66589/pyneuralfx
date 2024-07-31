import torch 
import torch.nn as nn 
from .base import * 

class SnapShotLSTM(SnapShot_Base):
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
            cell_type = 'lstm', 
            sample_rate = sample_rate, 
            layer_norm = layer_norm, 
            rnn_bias = rnn_bias
        )
    

class ConcatLSTM(Concat_Base):
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
            cell_type = 'lstm', 
            sample_rate = sample_rate, 
            num_conds = num_conds,
            layer_norm = layer_norm, 
            rnn_bias = rnn_bias
        )

class FiLMLSTM(FiLM_Base):
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
            cell_type = 'lstm', 
            sample_rate = sample_rate, 
            n_mlp_blocks = n_mlp_blocks, 
            mlp_size = mlp_size,
            num_conds = num_conds,
            layer_norm = layer_norm, 
            rnn_bias = rnn_bias,
            filmed = True
        )

class StaticHyperLSTM(StaticHyper_Base):
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
            cell_type = 'lstm', 
            sample_rate = sample_rate, 
            n_mlp_blocks = n_mlp_blocks, 
            mlp_size = mlp_size,
            num_conds = num_conds,
            layer_norm = layer_norm, 
            rnn_bias = rnn_bias
        )

class DynamicHyperLSTM(DynamicHyper_Base):
    def __init__(
        self,
        inp_channel: int,
        out_channel: int,
        rnn_size: int,
        sample_rate: int, 
        hyper_rnn_size: int = 8,
        n_z_size: int = 8,
        num_conds: int = 0,
        layer_norm: bool = False,
        rnn_bias: bool = True, 
    ):
        super().__init__(
            inp_channel = inp_channel, 
            out_channel = out_channel, 
            rnn_size = rnn_size, 
            cell_type = 'lstm', 
            sample_rate = sample_rate, 
            hyper_rnn_size = hyper_rnn_size, 
            n_z_size = n_z_size,
            num_conds = num_conds,
            layer_norm = layer_norm, 
            rnn_bias = rnn_bias
        )