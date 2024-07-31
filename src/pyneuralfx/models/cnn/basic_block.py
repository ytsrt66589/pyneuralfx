import torch 
import torch.nn as nn 


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, num_features: int):
        super().__init__()

        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = nn.Linear(cond_dim, num_features*2)

    def forward(self, x, cond):
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        
        g = g.unsqueeze(-1)
        b = b.unsqueeze(-1)
        
        x = self.bn(x)      # apply BatchNorm without affine 
        x = (x * g) + b     # then apply conditional affine

        return x
    

class HyperConv(nn.Module):
    def __init__(self,
                n_inp,
                n_output,
                kernel_size,
                dilation,
                cond_size,
                bias=False,):
        super().__init__()
        
        #arguments
        self.in_ch = n_inp
        self.out_ch = n_output
        self.kernel_size = kernel_size
        self.dilation = dilation 
        self.bias = bias

        self.kernel_subnet = nn.Sequential(
            nn.Linear(cond_size, self.in_ch, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.in_ch, self.in_ch * self.out_ch * kernel_size, bias=True)
        )

        self.bias = bias
        if self.bias:
            self.bias_subnet = nn.Sequential(
                nn.Linear(cond_size, self.in_ch, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(self.in_ch, self.out_ch, bias=True)
            )
        

    def forward(self, x, p):
        B = x.shape[0]
        tp = p.unsqueeze(-1)
        padding = self.dilation * (self.kernel_size - 1)
        start, end = padding, x.shape[-1]
        x = torch.cat([
            x[:, :, start-i*self.dilation:end-i*self.dilation] for i in range(self.kernel_size)
        ], dim=1)
        x = x.permute(0, 2, 1).contiguous().view(x.shape[0] * tp.shape[-1], x.shape[-1]//tp.shape[-1], x.shape[1])
        weight = self.kernel_subnet(p).view(B, self.in_ch * self.kernel_size, self.out_ch, tp.shape[-1]) # linear
        weight = weight.permute(0, 3, 1, 2).contiguous().view(B * tp.shape[-1], self.in_ch * self.kernel_size, self.out_ch)
        y = torch.bmm(x, weight)

        if self.bias:
            bias = self.bias_subnet(p).view(B, self.out_ch, tp.shape[-1])
            bias = bias.permute(0, 2, 1).contiguous().view(B * tp.shape[-1], self.out_ch)
            y = y + bias[:, None, :]
        y = y.view(B, -1, self.out_ch).permute(0, 2, 1).contiguous()
        return y