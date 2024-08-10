import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from typing import Callable, List, Optional, Tuple, Union, Dict, Any
from .utils import comlex_to_magnitude, DC_PreEmph

from ..transient_sep.tms_tools import tms_synthesis_tools
from .utils import comlex_to_magnitude, convert_tensor_to_numpy, loudness, crest_factor, rms_energy, STNSeparation, spectral_centroid
from ..loss_func.loss_func import (
    L1Loss,
    L2Loss,
    MRSTFTLoss,
    STFTLoss,
    ESRLoss
) 

#####################################################
######           Transient metrics             ######
#####################################################
class TransientPreservation(nn.Module):
    """Calculate the transient reconstruction loss (STFT loss) on the DCT domain 

    The algorithm is largely based on
    @article{verma2000extending,
        title={Extending spectral modeling synthesis with transient modeling synthesis},
        author={Verma, Tony S and Meng, Teresa HY},
        journal={Computer music journal},
        pages={47--59},
        year={2000},
        publisher={JSTOR}
    }

    Parameters
    ----------
    sr : int
        the sample rate of the target audio 
    """
    def __init__(self, sr: int):
        super().__init__()
        print('> [Metrics] --- Transient Loss ---')
        self.sr = sr
        self.loss_func = MRSTFTLoss()
        
    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        """Return transient reconstruction loss on the DCT domain by STFT loss 

        Parameters
        ----------
        predict : torch.Tensor
            predicted audio, shape: [batch_size, num_channel, audio_length]
        target : torch.Tensor
            target audio, shape: [batch_size, num_channel, audio_length]

        Returns
        -------
        torch.Tensor
            error value based on transient reconstruction 
        """
        min_len = min(predict.shape[-1], target.shape[-1])
        predict = predict[..., :min_len]
        target = target[..., :min_len]

        predict = convert_tensor_to_numpy(predict)
        target = convert_tensor_to_numpy(target)

        ### block by block (if no block by block, the algorithm is pretty slow)
        # to-do
        block_size = 4
        if min_len <= int(self.sr * block_size):
            pred_dct_transients = tms_synthesis_tools(predict, self.sr)
            target_dct_transients = tms_synthesis_tools(target, self.sr)
            #print('> pred_dct_transients: ', pred_dct_transients.shape)
        else:
            pred_dct_transients_list = []
            target_dct_transients_list = []
            st= 0 
            while (st + int(self.sr * block_size)) < min_len:
                sub_predict = predict[st:st + int(self.sr * block_size)]
                sub_target = target[st:st + int(self.sr * block_size)]
                sub_pred_dct_transients = tms_synthesis_tools(sub_predict, self.sr)
                sub_target_dct_transients = tms_synthesis_tools(sub_target, self.sr)
                pred_dct_transients_list.append(sub_pred_dct_transients)
                target_dct_transients_list.append(sub_target_dct_transients)
                st += int(self.sr * block_size)
            
            pred_dct_transients = np.concatenate(pred_dct_transients_list, axis=-1)
            target_dct_transients = np.concatenate(target_dct_transients_list, axis=-1)

        #pred_dct_transients = tms_synthesis_tools(predict, self.sr)
        #target_dct_transients = tms_synthesis_tools(target, self.sr)
        
        return self.loss_func(
            torch.from_numpy(pred_dct_transients), 
            torch.from_numpy(target_dct_transients)
        )

class TransientPreservation_v2(nn.Module):
    def __init__(self, sr: int):
        super().__init__()
        print('> [Metrics] --- Transient evaluation based on STN separation---')
        self.sr = sr
        self.stn_separation = STNSeparation(sr=sr)
        self.loss_func = ESRLoss()
        
    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        min_len = min(predict.shape[-1], target.shape[-1])
        predict = predict[..., :min_len]
        target = target[..., :min_len]

        _, predict_t, _ = self.stn_separation(predict)
        _, target_t, _ = self.stn_separation(target)

        return self.loss_func(
            predict_t, 
            target_t
        )


#######################################################
######           Level dynamics                  ###### 
#######################################################
# See: https://github.com/adobe-research/DeepAFx-ST/blob/main/deepafx_st/metrics.py
# > ================================================ < # 
# The following code is largely based on https://github.com/adobe-research/DeepAFx-ST/blob/main/deepafx_st/metrics.py
class LUFS(nn.Module):
    """LUFS: Calculate error based on audio loudness meter 

    This part of code is largely based on: https://github.com/csteinmetz1/pyloudnorm

    @inproceedings{steinmetz2021pyloudnorm,
        title={pyloudnorm: {A} simple yet flexible loudness meter in Python},
        author={Steinmetz, Christian J. and Reiss, Joshua D.},
        booktitle={150th AES Convention},
        year={2021}}
    
    Parameters
    ----------
    sr : int
       the sample rate of the audio 
    """
    def __init__(self,  sr: int = 48000):
        super().__init__()
        print('> [Metrics] --- LUFS ---')

        self.sample_rate = sr 
        
    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        """Return the LUFS error (currently supported mono only)

        Parameters
        ----------
        predict : torch.Tensor
            predicted audio, shape: [batch_size, num_channels, audio_length]
        target : torch.Tensor
            target audio, shape: [batch_size, num_channels, audio_length]

        Returns
        -------
        torch.Tensor
            error value based on loudness
        """
        return torch.nn.functional.l1_loss(
            loudness(predict.view(1, -1), self.sample_rate),
            loudness(target.view(1, -1), self.sample_rate),
        )

class CrestFactor(nn.Module): 
    """Calculate the error based on the crest factor (the peak value to the rms value) in units of dB
    """
    def __init__(self,):
        super().__init__()
        print('> [Metrics] --- Crest Factor (Dynamics range) [dB] ---')

    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.l1_loss(
            crest_factor(predict),
            crest_factor(target),
        )

class RMSEnergy(nn.Module):
    """Calculate the error based on the rms in units of dB
    """
    def __init__(self,):
        super().__init__()
        print('> [Metrics] --- Rms energy (Dynamics range) [dB] ---')

        
    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.l1_loss(
            rms_energy(predict),
            rms_energy(target),
        )
    
class SpectralCentroid(nn.Module):
    def __init__(self,):
        super().__init__()
        print('> [Metrics] --- Spectral Centroid ---')

        
    def forward(self, predict: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.l1_loss(
            spectral_centroid(predict),
            spectral_centroid(target),
        )