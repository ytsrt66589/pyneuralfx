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
from matplotlib import ticker
from .utils import * 
from .plotting import * 


def plot_distortion_curve(path_outdir, sr, gain, freq, feed_forward_func = None, *args, **kwargs):

    os.makedirs(path_outdir, exist_ok=True)
    path_song_distortion_curve = os.path.join(path_outdir, f'distortion-curve.png')

    n = np.arange(1000)
    x = gain * np.sin(2 * np.pi * n * freq / sr)
    _subx = x
    
    #nn_model = nn_model.to('cpu')
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to('cpu')
    
    ### forward 
    if feed_forward_func is None:
        y = feed_forward_func(x)
    else:
        y = feed_forward_func(x, *args, **kwargs)
    
    y = y.squeeze(0).squeeze(0).detach().numpy()

    # length alignment
    pre_room = len(_subx) - len(y)
    _subx = _subx[pre_room:]

    plt.plot(_subx, y)
    plt.xlabel('Input Gain')
    plt.ylabel('Output Gain')
    plt.tight_layout()
    plt.savefig(path_song_distortion_curve)
    plt.close()

def plot_harmonic_response(path_outdir, sr, gain, freq, feed_forward_func = None, *args, **kwargs):

    os.makedirs(path_outdir, exist_ok=True)
    path_song_harmonic_response = os.path.join(path_outdir, f'harmonic-response.png')

    num = 1000
    n = np.arange(num)
    x = gain * np.sin(2 * np.pi * n * freq / sr)
    _subx = x
    
    #nn_model = nn_model.to('cpu')
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to('cpu')
    
    ### forward 
    if feed_forward_func is None:
        y = feed_forward_func(x)
    else:
        y = feed_forward_func(x, *args, **kwargs)
    
    y = y.squeeze(0).squeeze(0).cpu().detach().numpy()

    # length alignment
    pre_room = len(_subx) - len(y)
    _subx = _subx[pre_room:]

    f = np.linspace(0, sr//2, num= num//2+1)
    H = normalize(np.fft.rfft(y))

    #plt.figure(figsize=(5, 3))

    plt.semilogx(f, 20 * np.log10(np.abs(H)))
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
    #plt.rcParams.update({'font.size': 14})
    
    plt.xlabel('Frequency [Hz]')#, fontsize=14
    plt.ylabel('Magnitude [dB]')#, fontsize=14
    # Increase tick label font size
    #plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.savefig(path_song_harmonic_response, dpi=600)
    plt.close()

def plot_sine_sweep_response_spec(path_outdir, sr, feed_forward_func = None, *args, **kwargs):

    os.makedirs(path_outdir, exist_ok=True)
    path_song_harmonic_response = os.path.join(path_outdir, f'sine-sweep-response-spec.png')

    num = sr * 5
    n = np.arange(num)
    f0 = 20 
    f1 = 20000
    beta = num / np.log(f1 / f0)
    phase = 2 * np.pi * beta * f0 * (pow(f1 / f0, n / num) - 1.0)
    phi = np.pi / 180
    x = np.cos((phase + phi)/sr)

    #x = gain * np.sin(2 * np.pi * n * freq / sr)
    _subx = x
    
    #nn_model = nn_model.to('cpu')
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float().to('cpu')
    
    ### forward 
    if feed_forward_func is None:
        y = feed_forward_func(x)
    else:
        y = feed_forward_func(x, *args, **kwargs)
    
    y = y.squeeze(0).squeeze(0).cpu().detach().numpy()

    # length alignment
    pre_room = len(_subx) - len(y)
    _subx = _subx[pre_room:]

    # S = np.abs(librosa.stft(_subx))
    # spec = librosa.amplitude_to_db(S, ref=np.max)
    # librosa.display.specshow(
    #     spec, 
    #     y_axis='linear', 
    #     #x_axis='time', 
    #     sr=sr
    # )

    #plt.figure(figsize=(5, 3))
    
    S = np.abs(librosa.stft(y))
    spec = librosa.amplitude_to_db(S, ref=np.max)
    librosa.display.specshow(
        spec, 
        y_axis='linear', 
        #x_axis='time', 
        sr=sr
    )

    #plt.title('Sine Sweep Response')
    
    #plt.rcParams.update({'font.size': 14})
    plt.tight_layout()
    plt.savefig(path_song_harmonic_response, dpi=600)
    plt.close()
    
