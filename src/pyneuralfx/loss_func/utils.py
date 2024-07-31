
import torch
import pyloudnorm as pyln
import scipy.signal as signal


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
    """Compute the crest factor of waveform.

    See: https://gist.github.com/endolith/359724

    """

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


