import torch 
import torch.nn as nn 
from .cells import * 
from ..base import RNN_Base
from .constants import RNNMAP


class SnapShot_Base(RNN_Base):
    def __init__(
        self,
        inp_channel: int,
        out_channel: int,
        rnn_size: int,
        cell_type: str, 
        sample_rate: int,
        layer_norm: bool = False,
        rnn_bias: bool = True, 
    ):
        super().__init__(sample_rate)

        # arguments 
        self.inp_channel = inp_channel
        self.out_channel = out_channel
        self.rnn_size = rnn_size
        self.cell_type = cell_type
        self.layer_norm = layer_norm
        self.rnn_bias = rnn_bias
        
        if self.layer_norm:
            self.main_rnn = LearnableCell(
                input_size = self.inp_channel, 
                hidden_size = self.rnn_size, 
                cell_type = self.cell_type,
                layer_norm = self.layer_norm,
                bias = self.rnn_bias
            )
        else:
            if self.cell_type == 'gru':
                self.main_rnn = torch.nn.GRU(
                    input_size = self.inp_channel, 
                    hidden_size = self.rnn_size, 
                    batch_first = True,
                    bias = self.rnn_bias
                )
            elif self.cell_type == 'lstm':
                self.main_rnn = torch.nn.LSTM(
                    input_size = self.inp_channel, 
                    hidden_size = self.rnn_size, 
                    batch_first = True,
                    bias = self.rnn_bias
                )
            elif self.cell_type == 'vanilla_rnn':
                self.main_rnn = torch.nn.RNN(
                    input_size = self.inp_channel, 
                    hidden_size = self.rnn_size, 
                    batch_first = True,
                    bias = self.rnn_bias
                )
        
        self.linear_out = nn.Linear(
            self.rnn_size,
            self.out_channel,
            bias = self.rnn_bias
        )

    def forward(self, x, c, h0):

        
        # B x C x T -> B x T x C
        x = x.permute(0, 2, 1)

        # for gru
        x, h = self.main_rnn(x, h0)
        x = self.linear_out(x)

        # out
        x = x.permute(0, 2, 1)

        return x, h, None


class Concat_Base(RNN_Base):
    def __init__(
        self,
        inp_channel: int,
        out_channel: int,
        rnn_size: int,
        cell_type: str, 
        sample_rate: int,
        num_conds: int = 0,
        layer_norm: bool = False,
        rnn_bias: bool = True, 
    ):
        super().__init__(sample_rate)

        # arguments 
        self.inp_channel = inp_channel
        self.out_channel = out_channel
        self.rnn_size = rnn_size
        self.cell_type = cell_type
        self.num_conds = num_conds
        self.layer_norm = layer_norm
        self.rnn_bias = rnn_bias
        
        if self.layer_norm:
            self.main_rnn = LearnableCell(
                input_size = self.inp_channel + self.num_conds, 
                hidden_size = self.rnn_size, 
                cell_type = self.cell_type,
                layer_norm = self.layer_norm,
                bias = self.rnn_bias
            )
        else:
            if self.cell_type == 'gru':
                self.main_rnn = torch.nn.GRU(
                    input_size = self.inp_channel + self.num_conds, 
                    hidden_size = self.rnn_size, 
                    batch_first = True,
                    bias = self.rnn_bias
                )
            elif self.cell_type == 'lstm':
                self.main_rnn = torch.nn.LSTM(
                    input_size = self.inp_channel + self.num_conds, 
                    hidden_size = self.rnn_size, 
                    batch_first = True,
                    bias = self.rnn_bias
                )
            elif self.cell_type == 'vanilla_rnn':
                self.main_rnn = torch.nn.RNN(
                    input_size = self.inp_channel + self.num_conds, 
                    hidden_size = self.rnn_size, 
                    batch_first = True,
                    bias = self.rnn_bias
                )
        
        self.linear_out = nn.Linear(
            self.rnn_size,
            self.out_channel,
            bias = self.rnn_bias
        )

    def forward(self, x, c, h0):

        if len(c.shape) == 2:
            c = c.unsqueeze(-1)
            c = c.expand(-1, -1, x.shape[-1])

        # cat x and cond
        if self.num_conds > 0:
            x = torch.cat((x, c), dim=1)
        
        # B x C x T -> B x T x C
        x = x.permute(0, 2, 1)

        # for gru
        x, h = self.main_rnn(x, h0)
        x = self.linear_out(x)

        # out
        x = x.permute(0, 2, 1)

        return x, h, None

class FiLM_Base(RNN_Base):
    def __init__(
        self,
        inp_channel: int,
        out_channel: int,
        rnn_size: int,
        cell_type: str, 
        sample_rate: int,
        num_conds: int = 0,
        mlp_size: int = 16,
        n_mlp_blocks: int = 2,
        layer_norm: bool = False,
        rnn_bias: bool = True, 
        filmed: bool = True
    ):
        super().__init__(sample_rate)

        if num_conds <= 0:
            raise ValueError(f'FiLM networks is used only when condition is provided')
        
        # arguments 
        self.inp_channel = inp_channel
        self.out_channel = out_channel
        self.rnn_size = rnn_size
        self.n_mlp_blocks = n_mlp_blocks
        self.mlp_size = mlp_size
        self.num_conds = num_conds
        self.layer_norm = layer_norm
        self.rnn_bias = rnn_bias
        self.cell_type = cell_type

        # MLP 
        self.SIZE_MAP = RNNMAP 

        self.gen = nn.ModuleList()
        for n in range(n_mlp_blocks):
            _input_features = num_conds if n == 0 else mlp_size
            self.gen.append(nn.Sequential(
                nn.Linear(_input_features, mlp_size, bias=True),
                nn.LeakyReLU(0.1)
            ))
        
        if n_mlp_blocks == 0:
            self.gen.append(nn.Linear(num_conds, rnn_size*(self.SIZE_MAP[self.cell_type]*4), bias=True))
        else:
            self.gen.append(nn.Linear(mlp_size, rnn_size*(self.SIZE_MAP[self.cell_type]*4), bias=True))
        
        
        self.gen = torch.nn.Sequential(*self.gen)

        self.main_rnn = LearnableCell(
            input_size = self.inp_channel, 
            hidden_size = self.rnn_size, 
            cell_type = self.cell_type,
            layer_norm = self.layer_norm,
            bias = self.rnn_bias,
            filmed = filmed
        )

        self.linear_out = nn.Linear(
            self.rnn_size,
            self.out_channel,
            bias = self.rnn_bias
        )
        
    def forward(self, x, c, h0):

        ih_ab, hh_ab = torch.chunk(self.gen(c), 2, dim=-1)
        a_i, b_i = torch.chunk(ih_ab, 2, dim=-1)
        a_h, b_h = torch.chunk(hh_ab, 2, dim=-1)

        
        x = x.permute(0, 2, 1)

        x, h = self.main_rnn(
            x, 
            h0,
            a_i,
            b_i,
            a_h,
            b_h
        )
    
        x = self.linear_out(x)
        
        x = x.permute(0, 2, 1)

        return x, h, None

class StaticHyper_Base(RNN_Base):
    def __init__(
        self,
        inp_channel: int,
        out_channel: int,
        rnn_size: int,
        cell_type: str, 
        sample_rate: int,
        n_mlp_blocks: int = 3,
        mlp_size: int = 8,
        num_conds: int = 0,
        layer_norm: bool = False,
        rnn_bias: bool = True, 
    ):
        super().__init__(sample_rate)

        if num_conds <= 0:
            raise ValueError(f'Static hyper networks is used only when condition is provided')
        
        # arguments 
        self.inp_channel = inp_channel
        self.out_channel = out_channel
        self.rnn_size = rnn_size
        self.n_mlp_blocks = n_mlp_blocks
        self.mlp_size = mlp_size
        self.num_conds = num_conds
        self.layer_norm = layer_norm
        self.rnn_bias = rnn_bias
        self.cell_type = cell_type
        

        # cells 
        self.main_rnn = NonlearnableCell(
            hidden_size = self.rnn_size, 
            cell_type = self.cell_type,
            layer_norm = self.layer_norm, 
        )

        self.linear_out = nn.Linear(
            self.rnn_size,
            self.out_channel,
            bias = self.rnn_bias
        )
        
        # MLP 
        self.SIZE_MAP = RNNMAP 
            

        if self.num_conds > 0: # if condition
            self.cond_mlp = nn.ModuleList()
            for n in range(self.n_mlp_blocks-1):
                _input_features = self.num_conds if n == 0 else self.mlp_size
                self.cond_mlp.append(nn.Sequential(
                    nn.Linear(_input_features, self.mlp_size, bias=True),
                    nn.LeakyReLU(0.1)
                ))
            self.cond_mlp = torch.nn.Sequential(*self.cond_mlp)

            # initialization 
            for idx, w in enumerate(self.cond_mlp.parameters()):
                if idx == 0:
                    bound = math.sqrt(self.SIZE_MAP[self.cell_type] / (self.mlp_size * self.num_conds))
                else:
                    bound = math.sqrt(self.SIZE_MAP[self.cell_type] / (self.mlp_size * self.mlp_size))
                w.data.uniform_(-bound, bound)


            _proj_ih_out = self.SIZE_MAP[self.cell_type] * self.inp_channel * self.rnn_size
            _proj_hh_out = self.SIZE_MAP[self.cell_type] * self.rnn_size * self.rnn_size
            _bias_out = self.SIZE_MAP[self.cell_type] * self.rnn_size

            self.proj_ih = torch.nn.Linear(
                self.mlp_size, 
                _proj_ih_out, 
                bias=True
            )
            
            self.proj_hh = torch.nn.Linear(
                self.mlp_size, 
                _proj_hh_out,
                bias = True
            )


            if self.rnn_bias: 
                self.proj_bih = torch.nn.Linear(
                    self.mlp_size,
                    _bias_out,
                    bias = True 
                )

                self.proj_bhh = torch.nn.Linear(
                    self.mlp_size,
                    _bias_out,
                    bias = True
                )
        

    def forward(self, x, c, h0):

        w = self.cond_mlp(c) 
        w_ih = self.proj_ih(w).reshape(-1, self.inp_channel, self.rnn_size * self.SIZE_MAP[self.cell_type])
        w_hh = self.proj_hh(w).reshape(-1, self.rnn_size, self.rnn_size * self.SIZE_MAP[self.cell_type])
        b_ih = None
        b_hh = None

        if self.rnn_bias:
            b_ih = self.proj_bih(w).reshape(-1, 1, self.rnn_size * self.SIZE_MAP[self.cell_type])
            b_hh = self.proj_bhh(w).reshape(-1, 1, self.rnn_size * self.SIZE_MAP[self.cell_type])
        
        x = x.permute(0, 2, 1)

        # core rnn
        x, h_out = self.main_rnn(x, h0, w_ih, w_hh, b_ih, b_hh)
        
        x = self.linear_out(x)
        
        x = x.permute(0, 2, 1)

        

        if self.cell_type == 'lstm':
            __h, __c = h_out
            return x, (__h.detach(), __c.detach()), (w_ih.detach(), w_hh.detach(), b_ih.detach(), b_hh.detach())
        
        return x, h_out.detach(), (w_ih.detach(), w_hh.detach(), b_ih.detach(), b_hh.detach())

class DynamicHyper_Base(RNN_Base):
    def __init__(
        self,
        inp_channel: int,
        out_channel: int,
        rnn_size: int,
        cell_type: str, 
        sample_rate: int,
        hyper_rnn_size: int = 8,
        n_z_size: int = 8,
        num_conds: int = 0,
        layer_norm: bool = False,
        rnn_bias: bool = True, 
    ):
        super().__init__(sample_rate)   

        # arguments 
        self.inp_channel = inp_channel
        self.out_channel = out_channel
        self.rnn_size = rnn_size
        self.hyper_rnn_size = hyper_rnn_size
        self.n_z_size = n_z_size
        self.num_conds = num_conds
        self.layer_norm = layer_norm
        self.rnn_bias = rnn_bias
        self.cell_type = cell_type

        self.main_rnn = DynamicHyperCell(
            input_size = self.num_conds,
            hidden_size = self.rnn_size, 
            cell_type = self.cell_type,
            hyper_size = self.hyper_rnn_size, 
            n_z = self.n_z_size, 
            layer_norm = self.layer_norm, 
            bias = rnn_bias
        )

        self.linear_out = nn.Linear(
            self.rnn_size,
            self.out_channel,
            bias = self.rnn_bias
        )
    

    def forward(
        self,
        x,
        c,
        h0,
        h_hat0,
    ):
        # B x C x T -> B x T x C
        x = x.permute(0, 2, 1)

        
        # for gru
        x, h = self.main_rnn(x, c, h0, h_hat0)
        
        x = self.linear_out(x)

        # out
        x = x.permute(0, 2, 1)

        return x, h, None