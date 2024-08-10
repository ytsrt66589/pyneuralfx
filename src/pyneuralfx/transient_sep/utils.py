import librosa
import numpy as np 
from scipy import ndimage


# > for fuzzy logic transient separation 
def stft(x, n_fft, hop_length, win_length):
    return librosa.stft(x, n_fft=n_fft, hop_length=hop_length, window='hann')

def istft(X, hop_length, win_length):
    return librosa.istft(X, hop_length=hop_length, window='hann')

def medfilt_vertical(abs_stft_buffer, n_median_v):
    kernel = np.zeros((n_median_v, 1))
    kernel[:, 0] = 1
    return ndimage.median_filter(abs_stft_buffer, footprint=kernel, mode='nearest')

def medfilt_horizontal(abs_stft_buffer, n_median_h):
    kernel = np.zeros((1, n_median_h))
    kernel[0, :] = 1
    return ndimage.median_filter(abs_stft_buffer, footprint=kernel, mode='nearest')



# for tms tools 

