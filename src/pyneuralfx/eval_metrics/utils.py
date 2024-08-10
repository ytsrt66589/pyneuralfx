
import torch
import pyloudnorm as pyln
import scipy.signal as signal
import torch.nn.functional as F

def convert_tensor_to_numpy(tensor, is_squeeze=True):
    """Utils functions for converting tensor to numpy 
    """
    if is_squeeze:
        tensor = tensor.squeeze()
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()

def comlex_to_magnitude(stft_complex):
    """Utils functions for converting complex spectrogram to maginitude 
    """
    stft_mag = torch.sum(stft_complex ** 2) ** 0.5
    return stft_mag

def loudness(x, sample_rate):
    # codes from : https://github.com/adobe-research/DeepAFx-ST/blob/main/deepafx_st/metrics.py
    """Compute the loudness in dB LUFS of waveform."""
    meter = pyln.Meter(sample_rate)

    # add stereo dim if needed
    if x.shape[0] < 2:
        x = x.repeat(2, 1)
    
    return torch.tensor(meter.integrated_loudness(x.permute(1, 0).numpy()))

def crest_factor(x):
    # codes from https://github.com/adobe-research/DeepAFx-ST/blob/main/deepafx_st/metrics.py
    """Compute the crest factor of waveform."""

    peak, _ = x.abs().max(dim=-1)
    rms = torch.sqrt((x ** 2).mean(dim=-1))

    return 20 * torch.log(peak / rms.clamp(1e-8))


def rms_energy(x):
    # codes from https://github.com/adobe-research/DeepAFx-ST/blob/main/deepafx_st/metrics.py
    """Compute the rms energy of waveform
    """
    rms = torch.sqrt((x ** 2).mean(dim=-1))

    return 20 * torch.log(rms.clamp(1e-8))

def spectral_centroid(x):

    spectrum = torch.fft.rfft(x).abs()
    normalized_spectrum = spectrum / spectrum.sum()
    normalized_frequencies = torch.linspace(0, 1, spectrum.shape[-1])
    spectral_centroid = torch.sum(normalized_frequencies * normalized_spectrum)

    return spectral_centroid

# > ================================================== <
#   High pass filter, for better reconstructing high-freq contents, proposed by VV
# > ================================================== <
class DC_PreEmph(torch.nn.Module):
    """
        code from: https://github.com/Alec-Wright/GreyBoxDRC/blob/main/loss_funcs.py
    """
    def __init__(self, R=0.995):
        super().__init__()

        t, ir = signal.dimpulse(signal.dlti([1, -1], [1, -R]), n=2000)
        ir = ir[0][:, 0]

        self.zPad = len(ir) - 1
        self.pars = torch.flipud(torch.tensor(ir, requires_grad=False, dtype=torch.FloatTensor.dtype)).unsqueeze(0).unsqueeze(0)
        
    def forward(
        self, 
        output: torch.tensor, 
        target: torch.tensor
    ):  
        # zero pad the input/target so the filtered signal is the same length
        output = torch.cat((torch.zeros(output.shape[0], 1, self.zPad).type_as(output), output), dim=2)
        target = torch.cat((torch.zeros(output.shape[0], 1, self.zPad).type_as(output), target), dim=2)

        output = torch.nn.functional.conv1d(output, self.pars.type_as(output), bias=None) # [B, 1, T]
        target = torch.nn.functional.conv1d(target, self.pars.type_as(output), bias=None) # [B, 1, T]

        return output, target



class STNSeparation(torch.nn.Module):
    def __init__(
        self, 
        sr,
        n_fft=4096, 
        hop_length=1024, 
        beta_u1=0.8, 
        beta_l1=0.7, 
        beta_u2=0.85, 
        beta_l2=0.75
    ):
        super().__init__()
        self.sr = sr 
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.beta_u1 = beta_u1
        self.beta_l1 = beta_l1
        self.beta_u2 = beta_u2
        self.beta_l2 = beta_l2

        self.L_h = max(3, int(round(0.2 * sr / hop_length)))
        if self.L_h % 2 != 1:
            self.L_h += 1
        self.L_v = max(3, int(round(500 * n_fft / sr)))
        if self.L_v % 2 != 1:
            self.L_v += 1

        self.n_fft_2 = 512
        self.hop_length_2 = 128
        self.L_h_2 = max(3, int(round(0.2 * sr / self.hop_length_2)))
        if self.L_h_2 % 2 != 1:
            self.L_h_2 += 1
        self.L_v_2 = max(3, int(round(500 * self.n_fft_2 / sr)))
        if self.L_v_2 % 2 != 1:
            self.L_v_2 += 1
        

    def forward(self, x):
        # x: [b, 1, len]
        # Step 1: STFT
        
        x = x.squeeze(1)
        X = torch.stft(
            x,
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.n_fft, 
            return_complex=True
        )
        X_mag = torch.abs(X)
        
        X_h = self.median_filter_time(X_mag, self.L_h)
        X_v = self.median_filter_freq(X_mag, self.L_v)
        
        R_s = X_h / (X_h + X_v + 1e-8)

        S = self.compute_masks(R_s, self.beta_u1, self.beta_l1)
        R = 1 - S

        X_s = S * X
        X_r = R * X

        x_s = torch.istft(X_s, n_fft=self.n_fft, hop_length=self.hop_length)
        x_r = torch.istft(X_r, n_fft=self.n_fft, hop_length=self.hop_length)

        X_r = torch.stft(
            x_r, 
            n_fft=self.n_fft_2, 
            hop_length=self.hop_length_2, 
            win_length=self.n_fft_2, 
            return_complex=True
        )
        X_abs_2 = torch.abs(X_r)

        X_h_2 = self.median_filter_time(X_abs_2, self.L_h_2)
        X_v_2 = self.median_filter_freq(X_abs_2, self.L_v_2)

        R_t = X_v_2 / (X_h_2 + X_v_2 + 1e-8)
        T = self.compute_masks(R_t, self.beta_u2, self.beta_l2)
        N = 1 - T

        X_t = T * X_r
        X_n = N * X_r


        x_t = torch.istft(X_t, n_fft=self.n_fft_2, hop_length=self.hop_length_2)
        x_n = torch.istft(X_n, n_fft=self.n_fft_2, hop_length=self.hop_length_2)

        # Ensure all outputs have the same length
        min_length = min(x_s.shape[-1], x_t.shape[-1], x_n.shape[-1])
        x_s = x_s[..., :min_length]
        x_t = x_t[..., :min_length]
        x_n = x_n[..., :min_length]

        return x_s, x_t, x_n

    def median_filter_time(self, x, median_h):
        # x shape: (batch, freq, time)
        pad = median_h // 2 
        x_padded = F.pad(x, (pad, pad), mode='reflect')
        x_unfolded = x_padded.unfold(-1, median_h, 1)
        x_filtered = torch.median(x_unfolded, dim=-1)[0]
        return x_filtered[..., :x.size(-1)]

    def median_filter_freq(self, x, median_v):
        # x shape: (batch, freq, time)
        pad = median_v // 2
        x_padded = F.pad(x.transpose(-1, -2), (pad, pad), mode='reflect').transpose(-1, -2)
        x_unfolded = x_padded.unfold(-2, median_v, 1)
        x_filtered = torch.median(x_unfolded, dim=-1)[0]
        return x_filtered[:, :x.size(1), :]

    def compute_masks(self, x, beta_u, beta_l):
        mask = torch.zeros_like(x)
        mask[x >= beta_u] = 1
        transition = (x >= beta_l) & (x < beta_u)
        mask[transition] = torch.sin(torch.pi/2 * (x[transition] - beta_l) / (beta_u - beta_l))**2
        return mask

