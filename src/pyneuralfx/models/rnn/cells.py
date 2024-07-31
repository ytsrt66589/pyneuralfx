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


# >>>>>>>>>>>>>>
class LearnableCell(nn.Module):
    def __init__(self, input_size: int , hidden_size: int, cell_type: str, layer_norm: bool = False, bias: bool = True, filmed: bool = False):
        super().__init__()

        # arg 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        self.layer_norm = layer_norm
        self.bias = bias 
        self.filmed = filmed

        # NN 
        self.x2h = nn.Linear(input_size, RNNMAP[cell_type] * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, RNNMAP[cell_type] * hidden_size, bias=bias)
        
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
        
        # initialize weights 
        self.initialize_weights()

    def forward(self, xs, state, alpha_i = None, beta_i = None, alpha_h = None, beta_h = None):
        
        outputs = []
        
        # match the shape 
        if self.cell_type == 'lstm':
            hx, cx = state 
            try:
                hx = rearrange(hx, '1 b h -> b 1 h')
                cx = rearrange(cx, '1 b h -> b 1 h')
            except:
                hx = hx
                cx = cx
            state = (hx, cx)
        else:
            try:
                state  = rearrange(state, '1 b h -> b 1 h')
            except:
                state = state 
            
        for i in range(xs.shape[1]):
            out, state = self.forward_cell(
                xs[:, i:i+1, ...],
                state,
                alpha_i, beta_i, alpha_h, beta_h)
            outputs += [out]
        return torch.cat(outputs, dim=1), state
    

    def forward_cell(self, x, h, alpha_i = None, beta_i = None, alpha_h = None, beta_h = None):

        if self.cell_type == 'lstm':
            hx, cx = h 
        else:
            hx = h 
        
        # forward single step
        if self.cell_type == 'gru':
            return self.forward_gru_cell(x, hx, alpha_i, beta_i, alpha_h, beta_h)
        elif self.cell_type == 'lstm':
            return self.forward_lstm_cell(x, (hx, cx), alpha_i, beta_i, alpha_h, beta_h)
        elif self.cell_type == 'vanilla_rnn':
            return self.forward_vanilla_rnn_cell(x, hx, alpha_i, beta_i, alpha_h, beta_h)
        else:
            raise ValueError(f'Detected invalid cell type: {self.cell_type}, please specify cell type as vanilla_rnn, lstm, or gru')
        
    def forward_gru_cell(self, x, h, alpha_i = None, beta_i = None, alpha_h = None, beta_h = None):
        
        ih = self.norm_i(self.x2h(x))
        hh = self.norm_h(self.h2h(h))

        if self.filmed:
            ih = (ih * alpha_i.unsqueeze(1)) + beta_i.unsqueeze(1)
            hh = (hh * alpha_h.unsqueeze(1)) + beta_h.unsqueeze(1)  

        i_r, i_i, i_n = ih.chunk(RNNMAP[self.cell_type], -1)
        h_r, h_i, h_n = hh.chunk(RNNMAP[self.cell_type], -1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate   = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (h - newgate)
        
        return hy, hy


    def forward_lstm_cell(self, x, h, alpha_i = None, beta_i = None, alpha_h = None, beta_h = None):
        # hidden state and cell state 
        hx, cx = h 

        if self.filmed:
            gates =  (self.norm_i(self.x2h(x)) * alpha_i.unsqueeze(1) + beta_i.unsqueeze(1)) + (self.norm_h(self.h2h(hx)) * alpha_h.unsqueeze(1) + beta_h.unsqueeze(1))
        else:
            gates =  self.norm_i(self.x2h(x)) + self.norm_h(self.h2h(hx))
        
        ingate, forgetgate, cellgate, outgate = gates.chunk(RNNMAP[self.cell_type], -1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        if self.norm_c is not None:
            cy = self.norm_c((forgetgate * cx) + (ingate * cellgate))

        hy = outgate * torch.tanh(cy)
        
        return hy, (hy, cy)

    def forward_vanilla_rnn_cell(self, x, h, alpha_i = None, beta_i = None, alpha_h = None, beta_h = None):
        
        if self.filmed:
            wx = self.norm_i(self.x2h(x)) * alpha_i.unsqueeze(1) + beta_i.unsqueeze(1)
            wh = self.norm_h(self.h2h(h)) * alpha_h.unsqueeze(1) + beta_h.unsqueeze(1)
        else:
            wx = self.norm_i(self.x2h(x))
            wh = self.norm_h(self.h2h(h))
        
        n_h = torch.tanh(wx + wh)

        #n_h = n_h.unsqueeze(1)

        return n_h, n_h
    
    def initialize_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class DynamicHyperCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, cell_type: str, hyper_size: int, n_z: int, layer_norm: bool = False, bias: bool = True):
        super().__init__()

        # arg 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hyper_size = hyper_size
        self.n_z = n_z
        self.cell_type = cell_type
        self.layer_norm = layer_norm
        self.bias = bias 


        self.hyper_rnn = LearnableCell(
            input_size=input_size + hidden_size, 
            hidden_size=hyper_size, 
            cell_type=cell_type,
            layer_norm=layer_norm, 
            bias=bias)

        # dynamic 
        self.z_h = nn.Linear(hyper_size, RNNMAP[cell_type] * n_z)
        self.z_x = nn.Linear(hyper_size, RNNMAP[cell_type] * n_z)
        self.z_b = nn.Linear(hyper_size, RNNMAP[cell_type] * n_z, bias=False)

        d_h = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(RNNMAP[cell_type])]
        self.d_h = nn.ModuleList(d_h)

        d_x = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(RNNMAP[cell_type])]
        self.d_x = nn.ModuleList(d_x)

        d_b = [nn.Linear(n_z, hidden_size) for _ in range(RNNMAP[cell_type])]
        self.d_b = nn.ModuleList(d_b)

        self.w_h = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, hidden_size)) for _ in range(RNNMAP[cell_type])])
        self.w_x = nn.ParameterList([nn.Parameter(torch.randn(hidden_size, input_size)) for _ in range(RNNMAP[cell_type])])


        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(RNNMAP[cell_type])])
            self.layer_norm_c = nn.LayerNorm(hidden_size)


    def forward(self, xs, cond, h_c, h_c_hat):
        
        outputs = []
        
        for i in range(xs.shape[1]):
            out, h_c, h_c_hat = self.forward_cell(
                xs[:, i:i+1, ...],
                cond,
                h_c,
                h_c_hat)
            outputs += [out.permute(1, 0, 2)]
        return torch.cat(outputs, dim=1), (h_c, h_c_hat)
    
    
    
    def forward_cell(self, x, cond, h_c, h_c_hat):

        # forward single step
        if self.cell_type == 'gru':
            return self.forward_gru_cell(x, cond, h_c, h_c_hat)
        elif self.cell_type == 'lstm':
            return self.forward_lstm_cell(x, cond, h_c, h_c_hat)
        elif self.cell_type == 'vanilla_rnn':
            return self.forward_vanilla_rnn_cell(x, cond, h_c, h_c_hat)
        else:
            raise ValueError(f'Detected invalid cell type: {self.cell_type}, please specify cell type as vanilla_rnn, lstm, or gru')

    def forward_gru_cell(self, x, cond, h_c, h_c_hat):

        h = h_c
        h_hat = h_c_hat 
        
        
        x_hat = torch.cat((h.squeeze(0), cond), dim=-1).unsqueeze(1)

        
        _, h_hat = self.hyper_rnn(x_hat, h_hat)

        h_hat = h_hat.squeeze(0)
        h = h.squeeze(0)
        x = x.squeeze(1)

        z_h = self.z_h(h_hat).chunk(RNNMAP[self.cell_type], dim=-1)
        z_x = self.z_x(h_hat).chunk(RNNMAP[self.cell_type], dim=-1)
        z_b = self.z_b(h_hat).chunk(RNNMAP[self.cell_type], dim=-1)

        
        rin = []



        for i in range(RNNMAP[self.cell_type]):
            d_h = self.d_h[i](z_h[i]).squeeze(1)
            d_x = self.d_x[i](z_x[i]).squeeze(1)
            if i == RNNMAP[self.cell_type]-1:
                y = torch.tanh(
                    rin[0] * d_h * torch.einsum('ij,bj->bi', self.w_h[i], h) + 
                    d_x * torch.einsum('ij,bj->bi', self.w_x[i], x) + 
                    self.d_b[i](z_b[i]).squeeze(1)
                )
            else:
                y = torch.sigmoid(
                    d_h * torch.einsum('ij,bj->bi', self.w_h[i], h) + d_x * torch.einsum('ij,bj->bi', self.w_x[i], x) + self.d_b[i](z_b[i]).squeeze(1)
                )
            if self.layer_norm:
                rin.append(self.layer_norm[i](y))
            else:
                rin.append(y)

        r, i, n = rin # [b, n]
        h_next = n + i * (h - n)
        h_next = h_next.unsqueeze(0)
        
        return h_next, h_next, h_hat
    
    def forward_lstm_cell(self, x, cond, h_c, h_c_hat):
        
        h, c = h_c
        h_hat, c_hat = h_c_hat 
        
        
        x_hat = torch.cat((h.squeeze(0), cond), dim=-1).unsqueeze(1)

        
        _, (h_hat, c_hat) = self.hyper_rnn(x_hat, (h_hat, c_hat))

        h_hat = h_hat.squeeze(0)
        c_hat = c_hat.squeeze(0)
        h = h.squeeze(0)
        c = c.squeeze(0)
        x = x.squeeze(1)

        z_h = self.z_h(h_hat).chunk(RNNMAP[self.cell_type], dim=-1)
        z_x = self.z_x(h_hat).chunk(RNNMAP[self.cell_type], dim=-1)
        z_b = self.z_b(h_hat).chunk(RNNMAP[self.cell_type], dim=-1)

        
        ifgo = []
        for i in range(RNNMAP[self.cell_type]):
            d_h = self.d_h[i](z_h[i]).squeeze(1)
            d_x = self.d_x[i](z_x[i]).squeeze(1)
            y = d_h * torch.einsum('ij,bj->bi', self.w_h[i], h) + d_x * torch.einsum('ij,bj->bi', self.w_x[i], x) + self.d_b[i](z_b[i]).squeeze(1)
            if self.layer_norm:
                ifgo.append(self.layer_norm[i](y))
            else:
                ifgo.append(y)

        i, f, g, o = ifgo
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

        if self.layer_norm:
            h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))
        else:
            h_next = torch.sigmoid(o) * torch.tanh(c_next)
        
        c_next = c_next.unsqueeze(0)
        h_next = h_next.unsqueeze(0)
        
        
        return h_next, (h_next, c_next), (h_hat, c_hat)

    def forward_vanilla_rnn_cell(self, x, c, h_c, h_c_hat):
        pass

    
    def initialize_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)