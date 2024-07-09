import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.utils import _single
import functools
import warnings

def erb_to_hz(x):
    """ 
        Convert ERB to Hz
        Args: 
            x (numpy.ndarray or float): Frequency in ERB scale
        Return: 
            numpy.ndarray or float: Frequency in Hz
    """
    return (np.exp(x/9.265)-1) * 24.7 * 9.265 

def hz_to_erb(x):
    '''Convert Hz to ERB

    Args:
        x (numpy.ndarray or float): Frequency in Hz

    Return:
        numpy.ndarray or float: Frequency in ERB scale
    '''
    return np.log(1+x/(24.7*9.265))*9.265


class ModulatedGaussianFilters(nn.Module):
    """ Modulated Gaussian filters

        The frequency response of this filter is given by

        [
            H(\omega) = e^{-(\omega-\omega_{c})^2/(2\sigma^2)} + e^{-(\omega+\omega_{c})^2/(2\sigma^2)}
        ]

        If one_sided is True, this frequency response is changed as

        [
            H(\omega) = e^{-(\omega-\omega_{c})^2/(2\sigma^2)}.
        ]

    """
    def __init__(self,
                n_filters,
                init_type = 'erb',
                min_bw = 1.0 * 2.0 * np.pi,
                initial_freq_range = [20.0, 48000/2],
                one_sided = False,
                init_sigma = 100.0 * 2.0 * np.pi,
                trainable = True):
        '''
        Args:
            n_filters (int): Number of filters
            init_type (str): Initialization type of center frequencies.
                If "erb", set them from initial_freq_range[0] to initial_freq_range[1] with an equal interval in the ERB scale.
                If "linear", set them from initial_freq_range[0] to initial_freq_range[1] with an equal interval in the linear frequency scale.
            min_bw (float): Minimum bandwidth in radian
            initial_freq_range ([float,float]): Initial frequency ranges in Hz, as tuple of minimum (typically 50) and maximum values (typically, half of Nyquist frequency)
            one_sided (bool): If True, ignore the term in the negative frequency region. If False, the corresponding impulse response is modulated Gaussian window.
            init_sigma (float): Initial value for sigma
            trainable (bool): Whether filter parameters are trainable or not.
        '''
        super().__init__()
        lf, hf = initial_freq_range

        if init_type == 'linear':
            mus = np.linspace(lf, hf, n_filters) * 2.0 * np.pi 
        elif init_type == 'erb':
            erb_mus = np.linspace(hz_to_erb(lf), hz_to_erb(hf), n_filters)
            mus = erb_to_hz(erb_mus) * 2.0 * np.pi
        
        sigma2s = init_sigma**2 * np.ones((n_filters,), dtype='f')
        
        self.min_ln_sigma2s = np.log(min_bw**2)
        self.mus = nn.Parameter(torch.from_numpy(mus).float())
        self._ln_sigma2s = nn.Parameter(torch.from_numpy(
            np.log(sigma2s)).float().clamp(min=self.min_ln_sigma2s)
        )

        self.phase = nn.Parameter(
            torch.zeros((n_filters,), dtype=torch.float)
        )
        self.phase.data.uniform_(0.0, np.pi)
        self.one_sided = one_sided

    @property
    def sigma2s(self):
        return self._ln_sigma2s.clamp(min=self.min_ln_sigma2s).exp()

    def get_frequency_responses(self, omega: torch.Tensor):
        """Sample frequency response at omega

        Args:
            omega (torch.Tensor): Angular frequencies (n_angs)
        
        Return
            tuple[torch.Tensor]: Real and imaginary parts of frequency responses sampled at omega.
        """
        if self.one_sided:
            resp_abs = torch.exp(
                -(omega[None,:] - self.mus[:,None]).pow(2.0)/(2.0*self.sigma2s[:,None])
            )
            resp_r = resp_abs * self.phase.cos()[:,None]
            resp_i = resp_abs * self.phase.sin()[:,None]
        else:
            resp_abs = torch.exp(-(omega[None,:] - self.mus[:,None]).pow(2.0)/(2.0*self.sigma2s[:,None])) # n_filters x n_angfreqs
            resp_abs2 = torch.exp(-(omega[None,:] + self.mus[:,None]).pow(2.0)/(2.0*self.sigma2s[:,None])) # to ensure filters whose impulse responses are real.
            resp_r = resp_abs * self.phase.cos()[:,None] + resp_abs2 * ((-self.phase).cos()[:,None])
            resp_i = resp_abs * self.phase.sin()[:,None] + resp_abs2 * ((-self.phase).sin()[:,None])
        return resp_r, resp_i
    
    def extra_repr(self):
        s = f'n_filters={int(self.mus.shape[0])}, one_sided={self.one_sided}'
        return s.format(**self.__dict__)

    @property
    def device(self):
        return self.mus.device
    

class TDModulatedGaussianFilters(ModulatedGaussianFilters):
    def __init__(self, 
                n_filters, 
                train_sample_rate, 
                init_type="erb", 
                min_bw=1.0*2.0*np.pi, 
                initial_freq_range=[50.0, 48000/2], 
                one_sided=False, 
                init_sigma=100.0*2.0*np.pi, 
                trainable=True):
        '''

        Args:
            n_filters (int): Number of filters
            train_sample_rate (float): Trained sampling frequency
            init_type (str): Initialization type of center frequencies.
                If "erb", set them from initial_freq_range[0] to initial_freq_range[1] with an equal interval in the ERB scale.
                If "linear", set them from initial_freq_range[0] to initial_freq_range[1] with an equal interval in the linear frequency scale.
            min_bw (float): Minimum bandwidth in radian
            initial_freq_range ([float,float]): Initial frequency ranges in Hz, as tuple of minimum (typically 50) and maximum values (typically, half of Nyquist frequency)
            one_sided (bool): If True, ignore the term in the negative frequency region. If False, the corresponding impulse response is modulated Gaussian window.
            init_sigma (float): Initial value for sigma
            trainable (bool): Whether filter parameters are trainable or not.
        '''
        super().__init__(n_filters=n_filters, init_type=init_type, min_bw=min_bw, initial_freq_range=initial_freq_range, one_sided=one_sided, init_sigma=init_sigma, trainable=trainable)
        self.register_buffer('train_sample_rate', torch.tensor(float(train_sample_rate)))

    def get_impulse_responses(self, sample_rate: int, tap_size: int):
        '''Sample impulse responses

        Args:
            sample_rate (int): Target sampling frequency
            tap_size (int): Tap size
        
        Return
            torch.Tensor: Sampled impulse responses (n_filters x tap_size)
        '''
        center_freqs_in_hz = self.mus/(2.0*np.pi)
        # check whether the center frequencies are below Nyquist rate
        if self.train_sample_rate > sample_rate:
            mask = center_freqs_in_hz <= sample_rate/2
        ###
        t = (torch.arange(0.0, tap_size, 1).type_as(center_freqs_in_hz)/sample_rate)
        t = (t - t.mean())[None,:]
        ###
        if self.one_sided:
            raise NotImplementedError
        else:
            c = 2.0*(2.0*np.pi*self.sigma2s[:,None]).sqrt()*(-self.sigma2s[:,None]*(t**2)/2.0).exp()
            filter_coeffs = c*(self.mus[:,None] @ t + self.phase[:,None]).cos() # n_filters x tap_size
        if self.train_sample_rate > sample_rate:
            filter_coeffs = filter_coeffs * mask[:,None]
        return filter_coeffs[:,torch.arange(tap_size-1,-1,-1)]



# common modules
def compute_Hilbert_transforms_of_filters(filters):
    '''Compute the Hilber transforms of the input filters

    Args:
        filters (torch.Tensor): Filters (n_filters x kernel_size)

    Return
        torch.Tensor: Hilbert transforms of the weights (out_channels x in_channels x kernel_size)
    '''
    ft_f = torch.fft.rfft(filters, n=filters.shape[1], dim=1, norm="ortho")
    hft_f = torch.view_as_complex(torch.stack((ft_f.imag, -ft_f.real), axis=-1))
    hft_f = torch.fft.irfft(hft_f, n=filters.shape[1], dim=1, norm="ortho")
    return hft_f.reshape(*(filters.shape))


class FreqRespSampConv1d(torch.nn.Module):
    '''SFI convolutional layer using the frequency domain filter design
    '''
    def __init__(self,
                in_channels,
                out_channels,
                n_samples,
                dilation,
                use_Hilbert_transforms=False):
        super().__init__()

        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.n_samples = n_samples
        self.sample_rate = None
        self.kernel_size = None
        self.stride = None
        self.padding = None
        self.dilation = dilation
        self.use_Hilbert_transforms = use_Hilbert_transforms
        
        ## 
        n_filters = in_channels * out_channels
        if self.use_Hilbert_transforms:
            if n_filters%2 == 1:
                raise ValueError(f'n_filters must be even when using Hilbert transforms of filters [n_filters={n_filters}]')
            n_filters //= 2
        self.n_filters = n_filters

        self._cache = dict()

    def prepare(self, sample_rate, kernel_size, stride, padding=0):
        self.sample_rate = sample_rate
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.padding = _single(padding)
        self.continuous_filters = ModulatedGaussianFilters(
            n_filters=self.n_filters,
            #train_sample_rate=self.sample_rate, 
            init_type="erb", 
            min_bw=1.0*2.0*np.pi, 
            initial_freq_range=[50.0, self.sample_rate/2], 
            one_sided=False, 
            init_sigma=100.0*2.0*np.pi, 
            trainable=True
        )

    def _compute_pinvW(self, device):
        kernel_size = self.kernel_size[0]
        sample_rate = self.sample_rate            
        P = (kernel_size-1)//2 if kernel_size%2 == 1 else kernel_size//2
        M = self.n_samples
        nyquist_rate = sample_rate / 2
        #
        ang_freqs = torch.linspace(0, nyquist_rate*2.0*np.pi, M).float().to(device)
        normalized_ang_freqs = ang_freqs / float(sample_rate)
        if kernel_size%2 == 1:
            seq_P = torch.arange(-P, P+1).float()[None,:].to(device)
            ln_W = -normalized_ang_freqs[:,None]*seq_P # M x 2P+1
        else:
            seq_P = torch.arange(-(P-1), P+1).float()[None,:].to(device)
            ln_W = -normalized_ang_freqs[:,None]*seq_P # M x 2P
        ln_W = ln_W.to(device)
        W = torch.cat((torch.cos(ln_W), torch.sin(ln_W)), dim=0) # 2*M x 2P
        ###
        pinvW = torch.pinverse(W) # 2P x 2M
        pinvW.requires_grad_(False)
        ang_freqs.requires_grad_(False)
        return ang_freqs, pinvW


    def approximate_by_FIR(self, device):
        '''Approximate frequency responses of analog filters with those of digital filters

        Args:
            device (torch.Device): Computation device
        
        Return:
            torch.Tensor: Time-reversed impulse responses of digital filters (n_filters x filter_degree (-P to P))
        '''
        
        cache_tag = (self.sample_rate, self.kernel_size, self.stride)
        if cache_tag in self._cache:
            ang_freqs, pinvW = self._cache[cache_tag]
            ang_freqs = ang_freqs.detach().to(device)
            pinvW = pinvW.detach().to(device)
        else:
            ang_freqs, pinvW = self._compute_pinvW(device)
            self._cache[cache_tag] = (ang_freqs.detach().cpu(), pinvW.detach().cpu())
        
        #cache_tag = (self.sample_rate, self.kernel_size, self.stride)
        #ang_freqs, pinvW = self._cache[cache_tag]
        #ang_freqs = ang_freqs.detach().to(device)
        #pinvW = pinvW.detach().to(device)

        ###
        resp_r, resp_i = self.continuous_filters.get_frequency_responses(ang_freqs) # n_filters x M
        resp = torch.cat((resp_r, resp_i), dim=1) # n_filters x 2M
        ###
        fir_coeffs = (pinvW[None,:,:] @ resp[:,:,None])[:,:,0] # n_filters x 2P
        return fir_coeffs[:,torch.arange(self.kernel_size[0]-1,-1,-1)] # time-reversed impulse response

    def weights(self, filters):
        '''Return weights

        Args:
            filters (torch.Tensor): Filters (n_channels x tap_size)

        Return:
            torch.Tensor: Weights. The shape is in_channel x out_channel x tap_size for an SFI convolutional layer
        '''
        filters = self.approximate_by_FIR(self.continuous_filters.device) # n_filters (or n_filters//2) x kernel_size

        if self.use_Hilbert_transforms:
            filters = torch.cat((filters, compute_Hilbert_transforms_of_filters(filters)), dim=0)
        
        return filters.reshape(self.out_channels, self.in_channels, -1)#.cuda()

    def forward(self, input):
        '''

        Args:
            input (torch.Tensor): Input feature (batch x in_channel x time)
        
        Return:
            torch.Tensor: Output feature (batch x out_channel x time)
        '''
        _weights = self.weights(self.continuous_filters)
        return F.conv1d(input, _weights, None, self.stride, self.padding, _single(self.dilation))

