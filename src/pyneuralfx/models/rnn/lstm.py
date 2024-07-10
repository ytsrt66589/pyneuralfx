import torch 
import torch.nn as nn 
from .base import * 



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
        super().__init__(inp_channel, out_channel, rnn_size, 'lstm', sample_rate, num_conds,layer_norm, rnn_bias)



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
        super().__init__(inp_channel, out_channel, rnn_size, 'lstm', sample_rate, n_mlp_blocks, mlp_size, num_conds,layer_norm, rnn_bias)