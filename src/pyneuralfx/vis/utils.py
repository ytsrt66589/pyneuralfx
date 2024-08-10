import sys
import os
import glob
import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.nn import functional as F

import librosa
import librosa.display

import matplotlib.pyplot as plt
from matplotlib import collections as mc


### Get Features
mag = lambda x: x[..., 0]**2 + x[...,1]**2

def get_spectrogram_librosa(wav, win_len=512, hop_len=64):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(wav, n_fft=win_len, hop_length=hop_len, center=False)))
    return D

def get_spectrum_librosa(signal, win_len, hop_len):
    spectrogram = np.abs(librosa.stft(signal, n_fft=win_len, hop_length=hop_len, center=False))
    spectrum = np.mean(spectrogram, axis=-1)
    return spectrum ** 0.5

def compute_loudness(signal, win_len, hop_len):
    def rms(sig):
        return np.sqrt(np.mean(sig ** 2))
    
    def mag2db(sig):
        return 20 * np.log10(sig + 10 ** -4)

    res = []
    len_sig = len(signal)
    for st in range(0, len_sig-win_len, hop_len):
        ed = min(len_sig, st + win_len)
        loudness_val = mag2db(rms(signal[st:ed]))
        res.append(loudness_val)
    return np.array(res)

def get_spectrum(tensor_in, win_len, hop_len, device='cpu'):
    window = torch.from_numpy(np.hanning(win_len)).float().to(device)
    stft = torch.stft(
                tensor_in, 
                window=window,
                hop_length=hop_len,
                n_fft=win_len,
                return_complex=False)
    spectrogram = mag(stft)
    spectrum = torch.mean(spectrogram ** 0.5, dim=-1)
    return spectrum ** 0.5

def get_smoothed_spectrum_diff(
        predict, 
        target, 
        win_size=4096,
        overlap=0.75,
        cutoff_freq_ratio=0.02, 
        weighting=True,
        scale=20,
        device='cpu'):
    
    hop_len = int((1 - overlap) * win_size)
    predict = predict.squeeze()
    target = target.squeeze()

    # compute spectrum
    spec_pred = get_spectrum(
            predict,
            win_len=win_size,
            hop_len=hop_len,
            device=device).squeeze() / win_size
    spec_anno = get_spectrum(
            target, 
            win_len=win_size,
            hop_len=hop_len,
            device=device).squeeze() / win_size
    # difference
    spec_diff = scale * (torch.log10(spec_anno) - torch.log10(spec_pred))
    return spec_diff

def normalize(x):
    return x / np.max(np.abs(x))
