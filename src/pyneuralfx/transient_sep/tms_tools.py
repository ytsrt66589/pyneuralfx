'''
    This code is highly inspired and referenced by: sms-tools 
    https://github.com/MTG/sms-tools 
'''
import numpy as np 

from .sms_tools import utilFunctions as UF
from .sms_tools import sineModel as SM
from .sms_tools import stochasticModel as STM

from scipy.signal import get_window
import torch_dct as dct
import torch
import time

def get_config():
    window ='hann'
    M = 2048
    N = 2048 
    t = -90
    minSineDur = 0.001
    nH = 60
    minf0 = 60
    maxf0 = 8000
    f0et = 7
    harmDevSlope = 0.01
    stocf = 0.04 #0.04
    
    maxnSines = 256
    freqDevOffset = 20
    freqDevSlope = 0.001
    
    # no need to modify anything after this
    Ns = 512
    H = 128
    return {
        'sms_sm':{
            'window': window,
            'M': M,
            'N': N,
            't': t,
            'minSineDur': minSineDur,
            'maxnSines': maxnSines,
            'freqDevOffset': freqDevOffset,
            'freqDevSlope': freqDevSlope,
            'stocf': stocf,
            'Ns': Ns,
            'H': H
        },
        'tms_transient_sm':{
            'window': window,
            'M': 512,
            'N': 512,
            't': t,
            'minSineDur': 0.00003, 
            'maxnSines': maxnSines,
            'freqDevOffset': freqDevOffset,
            'freqDevSlope': freqDevSlope,
            'Ns': Ns,
            'H': H
        }, 
        'tms_transient_noise':{
            'H': H,
            'stocf': stocf,
        }, 
    }


def transient_synth(transient_signal, fs, config, transient_type, dct_blocksize):
    """Transient synthesis algorithm based on the sinusoidal analysis/synthesis methods on the DCT domain
    """
    transient_results = []
    dct_transient_results = []
    start, end = 0, int(fs * dct_blocksize)
    while start < len(transient_signal):
        transient1 = transient_signal[start:end] 
        transient1_dct = dct.dct(torch.from_numpy(transient1), "ortho").numpy() # turn to dct domain
        # Sin Anal & Synth (DCT domain)
        w = get_window(config[transient_type]['window'], config[transient_type]['M'], fftbins=True)
        tfreq, tmag, tphase = SM.sineModelAnal(
                transient1_dct, 
                fs, 
                w, 
                config[transient_type]['N'], 
                config[transient_type]['H'], 
                config[transient_type]['t'], 
                config[transient_type]['maxnSines'], 
                config[transient_type]['minSineDur'], 
                config[transient_type]['freqDevOffset'],
                config[transient_type]['freqDevSlope'])
        tms_transient1_dct = SM.sineModelSynth(tfreq, tmag, tphase, config[transient_type]['Ns'], config[transient_type]['H'], fs)
        dct_transient_results.append(tms_transient1_dct)
        # IDCT to time domain
        tms_transient1 = dct.idct(torch.from_numpy(tms_transient1_dct), "ortho").numpy()
        transient_results.append(tms_transient1)
        
        # next frame (no overlap)
        start = int(end)
        end = start + int(fs * dct_blocksize)
        if end >= len(transient_signal):
            end = len(transient_signal)
    return np.concatenate(transient_results, 0).astype(np.float32).reshape(-1), np.concatenate(dct_transient_results, 0).astype(np.float32).reshape(-1), transient_results, dct_transient_results


def tms_synthesis_tools(wav, fs):
    harmonic_type = 'sms_sm'
    transient_type = 'tms_transient_sm'
    noise_type = 'tms_transient_noise'
    config = get_config()
    #print('Harmonic ')
    st_time = time.time()
    # harmonic 
    
    w = get_window(config[harmonic_type]['window'], config[harmonic_type]['M'], fftbins=True)
    tfreq, tmag, tphase = SM.sineModelAnal(
        wav, 
        fs, 
        w, 
        config[harmonic_type]['N'], 
        config[harmonic_type]['H'], 
        config[harmonic_type]['t'], 
        config[harmonic_type]['maxnSines'], 
        config[harmonic_type]['minSineDur'], 
        config[harmonic_type]['freqDevOffset'],
        config[harmonic_type]['freqDevSlope'])
    tms_harmonic = SM.sineModelSynth(tfreq, tmag, tphase, config[harmonic_type]['Ns'], config[harmonic_type]['H'], fs)
    ed_time = time.time()
    #print('Harmonics: ', ed_time-st_time)

    #print('Transient')
    st_time = time.time()
    #print('tms_harmonic: ', tms_harmonic.shape)
    # transient 
    min_len = min([len(tms_harmonic), len(wav)])
    tms_first_residual = UF.sineSubtraction(wav, config[harmonic_type]['Ns'], config[harmonic_type]['H'], tfreq, tmag, tphase, fs)  #wav[:min_len] - tms_harmonic[:min_len]
    tms_first_residual = wav[:min_len] - tms_harmonic[:min_len]

    # for long wavs
    dct_blocksize = 2 #sec
    tms_transient, tms_dct_transient, transients, dct_transients = transient_synth(tms_first_residual, fs, config, transient_type, dct_blocksize)
    ed_time = time.time()
    #print('Transients: ', ed_time-st_time)
    return tms_dct_transient



