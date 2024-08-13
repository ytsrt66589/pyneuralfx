import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
from .utils import comlex_to_magnitude, DC_PreEmph

#################################################################
######          Base Class of the Loss Function            ######
#################################################################
class _Loss_Base(torch.nn.Module):
    def __init__(self, pre_emp: bool = False):
        super().__init__()

        self.pre_emp_filter = None
        if pre_emp:
            self.pre_emp_filter = DC_PreEmph()


    def forward(self, predict: torch.tensor, target: torch.tensor): 
        pass 

#########################################
######          L1 loss            ######
#########################################
class L1Loss(_Loss_Base):
    def __init__(self, pre_emp: bool = False):
        super().__init__(pre_emp)
        print('> [Loss] --- Temporal L1 Loss ---')

    def forward(
        self, 
        predict: torch.tensor, 
        target: torch.tensor
    ):
        if self.pre_emp_filter:
            predict, target = self.pre_emp_filter(predict, target)
        return F.l1_loss(predict, target)


#########################################
######          L2 loss            ######
#########################################
class L2Loss(_Loss_Base):
    def __init__(self, pre_emp: bool = False):
        super().__init__(pre_emp)
        print('> [Loss] --- Temporal L2 Loss ---')

    def forward(
        self, 
        predict: torch.tensor, 
        target: torch.tensor
    ):
        if self.pre_emp_filter:
            predict, target = self.pre_emp_filter(predict, target)
        return torch.mean((predict - target).pow(2))


#########################################
######           STFT              ######
#########################################
class STFTLoss(_Loss_Base):
    def __init__(
        self, 
        win_len: int = 2048, 
        overlap: float = 0.75, 
        is_complex: bool = True,
        pre_emp: bool = False
    ):
        super().__init__(pre_emp)
        print('> [Loss] --- STFT Complex Loss ---')
        print('> [Loss] is complex:', is_complex)
        print('> [Loss] win_len: {}, overlap: {}'.format(win_len, overlap))

        self.win_len     = win_len
        self.is_complex = is_complex
        self.hop_len     = int((1-overlap)*win_len)

        self.window = nn.Parameter(
                torch.from_numpy(
                    np.hanning(win_len)
                ).float(), requires_grad=False)

    def forward(
        self,
        predict: torch.tensor, 
        target: torch.tensor
    ):
        # (B, C, T)  to (B x C, T)
        if self.pre_emp_filter:
            predict, target = self.pre_emp_filter(predict, target)

        predict = predict.reshape(-1, predict.shape[-1])
        target  = target.reshape( -1, target.shape[-1])

        # stft - predict
        stft_predict = torch.stft(
                predict, 
                n_fft=self.win_len, 
                window=self.window, 
                hop_length=self.hop_len, 
                return_complex=True,
                center=False)

        # stft - target
        stft_target = torch.stft(
                target, 
                n_fft=self.win_len, 
                window=self.window, 
                hop_length=self.hop_len, 
                return_complex=True,
                center=False)

        # loss 
        if self.is_complex:
            loss_final = F.l1_loss(stft_predict, stft_target)
        else:
           stft_predict_mag = comlex_to_magnitude(stft_predict)
           stft_target_mag  = comlex_to_magnitude(stft_target)
           loss_final = F.l1_loss(stft_predict_mag, stft_target_mag)

        return loss_final


#########################################
######  Multi-resolution STFT      ######
#########################################
class MRSTFTCLoss(_Loss_Base):
    def __init__(
        self,
        scales: List = [8192, 2048, 512, 128],
        overlap: float = 0.75, 
        pre_emp: bool = False
    ):
        super().__init__(pre_emp)
        print('> [Loss] --- Multi-resolution STFT Loss ---')
    
        self.scales = scales
        self.overlap = overlap
        self.num_scales = len(self.scales)

        self.windows = nn.ParameterList(
            nn.Parameter(torch.from_numpy(np.hanning(scale)).float(), requires_grad=False) for scale in self.scales
        )

    def forward(
        self,
        predict: torch.tensor, 
        target: torch.tensor
    ):
        if self.pre_emp_filter:
            predict, target = self.pre_emp_filter(predict, target)

        # (B, C, T)  to (B x C, T)
        x      = predict.reshape(-1, predict.shape[-1])
        x_orig = target.reshape( -1, target.shape[-1])

        amp = lambda x: x[:,:,:,0]**2 + x[:,:,:,1]**2

        stfts = []
        for i, scale in enumerate(self.scales):
            cur_fft = torch.stft(x, n_fft=scale, window=self.windows[i], hop_length=int((1-self.overlap)*scale), center=False, return_complex=True)
            stfts.append(cur_fft)

        stfts_orig = []
        for i, scale in enumerate(self.scales):
            cur_fft = torch.stft(x_orig, n_fft=scale, window=self.windows[i], hop_length=int((1-self.overlap)*scale), center=False, return_complex=True)
            stfts_orig.append(cur_fft)

        # Compute loss scale x batch
        lin_loss_final = 0
        #log_loss_final = 0
        for i in range(self.num_scales):
            lin_loss = torch.mean(abs(stfts_orig[i] - stfts[i]))
            #log_loss = torch.mean(abs(torch.log(stfts_orig[i] + 1e-4) - torch.log(stfts[i] + 1e-4)))  

            lin_loss_final += lin_loss
            #log_loss_final += log_loss
        
        lin_loss_final /= self.num_scales
        #log_loss_final /= self.num_scales

        return lin_loss_final #+ log_loss_final

#########################################
######         MRSTFT loss         ######
#########################################
class MRSTFTLoss(_Loss_Base):
    def __init__(
        self,
        scales: List = [2048, 512, 128, 32],
        overlap: float = 0.75, 
        pre_emp: bool = False
    ):
        super().__init__(pre_emp)
        print('> [Loss] --- Multi-resolution STFT Loss ---')
    
        self.scales = scales
        self.overlap = overlap
        self.num_scales = len(self.scales)

        self.windows = nn.ParameterList(
            nn.Parameter(torch.from_numpy(np.hanning(scale)).float(), requires_grad=False) for scale in self.scales
        )

    def forward(
        self,
        predict: torch.tensor, 
        target: torch.tensor
    ):
        if self.pre_emp_filter:
            predict, target = self.pre_emp_filter(predict, target)

        # (B, C, T)  to (B x C, T)
        x      = predict.reshape(-1, predict.shape[-1])
        x_orig = target.reshape( -1, target.shape[-1])

        amp = lambda x: x[:,:,:,0]**2 + x[:,:,:,1]**2

        stfts = []
        for i, scale in enumerate(self.scales):
            cur_fft = torch.stft(x, n_fft=scale, window=self.windows[i], hop_length=int((1-self.overlap)*scale), center=False, return_complex=False)
            stfts.append(amp(cur_fft))

        stfts_orig = []
        for i, scale in enumerate(self.scales):
            cur_fft = torch.stft(x_orig, n_fft=scale, window=self.windows[i], hop_length=int((1-self.overlap)*scale), center=False, return_complex=False)
            stfts_orig.append(amp(cur_fft))

        # Compute loss scale x batch
        lin_loss_final = 0
        log_loss_final = 0
        for i in range(self.num_scales):
            lin_loss = torch.mean(abs(stfts_orig[i] - stfts[i]))
            log_loss = torch.mean(abs(torch.log(stfts_orig[i] + 1e-4) - torch.log(stfts[i] + 1e-4)))  

            lin_loss_final += lin_loss
            log_loss_final += log_loss
        
        lin_loss_final /= self.num_scales
        log_loss_final /= self.num_scales

        return lin_loss_final + log_loss_final


#########################################
######           ESR               ######
#########################################
class ESRLoss(_Loss_Base):
    def __init__(self, pre_emp: bool = False):
        super().__init__(pre_emp)
        print('> [Loss] --- ESR Loss ---')

        self.epsilon = 1e-12
    def forward(
        self, 
        predict: torch.tensor, 
        target: torch.tensor
    ):
        if self.pre_emp_filter:
            predict, target = self.pre_emp_filter(predict, target)

        y_pred = predict
        y = target

        loss = torch.add(y, -y_pred)
        loss = torch.pow(loss, 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(y, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss


#########################################################
######           Eliminating DC Bias               ######
#########################################################
class DCLoss(_Loss_Base):
    def __init__(self, pre_emp: bool = False):
        super().__init__(pre_emp)
        self.epsilon = 0.00001
        
    def forward(
        self, 
        predict: torch.tensor, 
        target: torch.tensor
    ):
        if self.pre_emp_filter:
            predict, target = self.pre_emp_filter(predict, target)

        loss = torch.pow(torch.add(torch.mean(target, 0), -torch.mean(predict, 0)), 2)
        loss = torch.mean(loss)
        energy = torch.mean(torch.pow(target, 2)) + self.epsilon
        loss = torch.div(loss, energy)
        return loss



# for development easily 
class HybridLoss(torch.nn.Module):
    def __init__(self, pre_emp: bool = False):
        super().__init__()
        print('> [Loss] --- Hybrid Trans Loss ---')

        self.mae = L1Loss(pre_emp)
        self.mrstft = MRSTFTLoss(pre_emp=pre_emp)

    def forward(
        self, 
        predict: torch.tensor, 
        target: torch.tensor
    ):  
        return self.mae(predict, target) + 0.1*self.mrstft(predict, target) 

