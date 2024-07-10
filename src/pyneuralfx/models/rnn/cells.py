import math 
import torch 
import torch.nn as nn 
from einops import rearrange
from .constants import RNNMAP




class NonlearnableCell(nn.Module): 
    def __init__(self, hidden_size: int, cell_type: str, layer_norm: bool = False):
        super().__init__()

        if cell_type not in ['vanilla_rnn', 'lstm', 'gru']:
            raise ValueError(f'Invalid cell type: {cell_type}, please specify cell type as vanilla_rnn, lstm, or gru')
        

        # layer norm 
        self.norm_i, self.norm_h, self.norm_c = None, None, None
        if cell_type == 'gru':
            self.norm_i = nn.LayerNorm(RNNMAP[cell_type] * hidden_size, elementwise_affine = False) if layer_norm else nn.Identity()
            self.norm_h = nn.LayerNorm(RNNMAP[cell_type] * hidden_size, elementwise_affine = False) if layer_norm else nn.Identity()
        elif cell_type == 'lstm':
            self.norm_i = nn.LayerNorm(RNNMAP[cell_type] * hidden_size, elementwise_affine = False) if layer_norm else nn.Identity()
            self.norm_h = nn.LayerNorm(RNNMAP[cell_type] * hidden_size, elementwise_affine = False) if layer_norm else nn.Identity()
            self.norm_c = nn.LayerNorm(hidden_size, elementwise_affine = False) if layer_norm else nn.Identity()
        elif cell_type == 'vanilla_rnn':
            self.norm_i = nn.LayerNorm(hidden_size, elementwise_affine = False) if layer_norm else nn.Identity()
            self.norm_h = nn.LayerNorm(hidden_size, elementwise_affine = False) if layer_norm else nn.Identity()

        self.cell_type = cell_type 

    def forward_cell(self, x, h, w_ih, w_hh, b_ih = None, b_hh = None):

        if self.cell_type == 'lstm':
            hx, cx = h 
        else:
            hx = h 
        
        # matrix multiplication 
        ih = torch.bmm(x, w_ih) 
        hh = torch.bmm(hx, w_hh)
        if b_ih is not None and b_hh is not None:
            ih = ih + b_ih
            hh = hh + b_hh

        # normalization or identity 
        ih = self.norm_i(ih)
        hh = self.norm_h(hh)

        # forward single step
        if self.cell_type == 'gru':
            return self.forward_gru_cell(hx, ih, hh)
        elif self.cell_type == 'lstm':
            return self.forward_lstm_cell(cx, ih, hh)
        elif self.cell_type == 'vanilla_rnn':
            return self.forward_vanilla_rnn_cell(ih, hh)
        else:
            raise ValueError(f'Detected invalid cell type: {self.cell_type}, please specify cell type as vanilla_rnn, lstm, or gru')
        
    def forward_gru_cell(self, hx, ih, hh):
        i_r, i_i, i_n = ih.chunk(RNNMAP[self.cell_type], -1)
        h_r, h_i, h_n = hh.chunk(RNNMAP[self.cell_type], -1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate   = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hx - newgate)

        return hy, hy
    
    def forward_lstm_cell(self, cx, ih, hh):
        gates = (ih + hh)
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(RNNMAP[self.cell_type], -1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        if self.norm_c is not None:
            cy = self.norm_c((forgetgate * cx) + (ingate * cellgate))

        hy = outgate * torch.tanh(cy)
        
        return hy, (hy, cy)
    
    def forward_vanilla_rnn_cell(self, ih, hh):
        nh = torch.tanh(ih + hh)
        return nh, nh
    
    def forward(self, xs, state, w_ih, w_hh, b_ih, b_hh):
        outputs = []
        
        # match the shape 
        if self.cell_type == 'lstm':
            hx, cx = state 
            hx = rearrange(hx, '1 b h -> b 1 h')
            cx = rearrange(cx, '1 b h -> b 1 h')
            state = (hx, cx)
        else:
            state  = rearrange(state, '1 b h -> b 1 h')

        for i in range(xs.shape[1]):
            out, state = self.forward_cell(
                xs[:, i:i+1, ...],
                state,
                w_ih,
                w_hh,
                b_ih,
                b_hh)
            outputs += [out]
        return torch.cat(outputs, dim=1), state



class LearnableCell(nn.Module):
    def __init__(self, input_size: int , hidden_size: int, cell_type: str, layer_norm: bool = False, bias: bool = True):
        super().__init__()

        # arg 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.layer_norm = layer_norm
        self.bias = bias 



    def forward(self):
        pass 

    def forward_cell(self):
        pass 

    
    def init_weights(self):
        pass 