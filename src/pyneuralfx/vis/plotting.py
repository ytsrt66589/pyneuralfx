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
from .utils import * 

def plot_waveform(wav, sr, filename=None, title=''):
    librosa.display.waveshow(wav, sr=sr)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.ylim([-1.0, 1.0])
    if filename:
        plt.savefig(filename)
        plt.close()
         
def plot_spec(spec, sr, hop_length, filename=None, title=''):
    librosa.display.specshow(
        spec, 
        y_axis='linear', 
        x_axis='time', 
        sr=sr, 
        hop_length=hop_length,
        cmap='magma')
    plt.title(title)
    if filename:
        plt.savefig(filename)
        plt.close()

def plot_spec_diff(path_y_anno, path_y_pred, path_outdir, sr, num_channel):

    os.makedirs(path_outdir, exist_ok=True)

    # load
    y_anno, sr_anno = sf.read(path_y_anno)
    y_pred, sr_pred = sf.read(path_y_pred)

    assert sr_anno == sr 
    assert sr_pred == sr 

    if num_channel == 1:
        y_anno = y_anno[..., None]
        y_pred = y_pred[..., None]

    # length alignment
    min_len = min(len(y_anno), len(y_pred))
    y_anno = y_anno[:min_len]
    y_pred = y_pred[:min_len]

    # plot spec_diff
    for cidx in range(num_channel):
        FREQ_LIM =(20, sr//2)
        f_axis = np.linspace(0.0, sr/2, 4096//2 + 1)
        plt.figure(figsize=(10, 5))
        plt.xscale('log')

        path_song_spec_diff = os.path.join(path_outdir, f'spec_diff_ch-{cidx}.png')
        y_anno_t = torch.from_numpy(y_anno[:, cidx])
        y_pred_t = torch.from_numpy(y_pred[:, cidx])
        spec_diff = get_smoothed_spectrum_diff(y_pred_t, y_anno_t)
        
        plt.plot(f_axis, spec_diff)

        plt.grid(visible=True, which='major', color='#515A62', linestyle='-')
        plt.grid(visible=True, which='minor', color='grey', linestyle=':')
        plt.xlim(FREQ_LIM)
        plt.minorticks_on()
        plt.xlabel('Frequency', fontsize=14)
        plt.ylabel('Magnitude (dB)', fontsize=14)
        plt.title('Spectrum Difference')
        plt.tight_layout()
        plt.savefig(path_song_spec_diff)
        plt.close()

def plot_wav_displayment(path_y_anno, path_y_pred, path_outdir, sr, num_channel):
    os.makedirs(path_outdir, exist_ok=True)

    # load
    y_anno, sr_anno = sf.read(path_y_anno)
    y_pred, sr_pred = sf.read(path_y_pred)

    assert sr_anno == sr 
    assert sr_pred == sr 

    
    if num_channel == 1:
        y_anno = y_anno[..., None]
        y_pred = y_pred[..., None]

    # length alignment
    min_len = min(len(y_anno), len(y_pred))
    y_anno = y_anno[:min_len]
    y_pred = y_pred[:min_len]

    # start = 20
    # y_anno = y_anno[sr_anno*(start):sr_anno*(start+1)]
    # y_pred = y_pred[sr_anno*(start):sr_pred*(start+1)]
    
    # plot wav_diff
    for cidx in range(num_channel):
        path_song_wav_diff = os.path.join(path_outdir, f'wav_ch-{cidx}.png')

        librosa.display.waveshow(y_anno[:, cidx], sr=sr, label='Target')
        librosa.display.waveshow(y_pred[:, cidx], sr=sr, alpha = 0.5, label='Predict')
        plt.title('Wav diff')
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.ylim([-1.0, 1.0])
        plt.tight_layout()
        plt.legend()
        plt.savefig(path_song_wav_diff)
        plt.close()





