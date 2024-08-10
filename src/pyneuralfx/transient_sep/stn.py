from .utils import * 


def compute_masks(R, beta_u, beta_l):
    mask = np.zeros_like(R)
    mask[R >= beta_u] = 1
    transition = np.logical_and(beta_l <= R, R < beta_u)
    mask[transition] = np.sin(np.pi/2 * (R[transition] - beta_l) / (beta_u - beta_l)) ** 2
    return mask

def stn_decomposition(x, sr, n_fft=8192, hop_length=2048, beta_u_1=0.8, beta_l_1=0.7, beta_u_2=0.85, beta_l_2=0.75):
    # Stage 1: Sines vs Residual
    X = stft(x, n_fft, hop_length, n_fft)
    
    L_h = max(3, int(round(0.2 * sr / hop_length)))
    L_v = max(3, int(round(500 * n_fft / sr)))
    
    X_abs = np.abs(X)
    X_h = medfilt_horizontal(X_abs, L_h)
    X_v = medfilt_vertical(X_abs, L_v)
    
    R_s = X_h / (X_h + X_v + 1e-8)
    
    S = compute_masks(R_s, beta_u_1, beta_l_1)
    R = 1 - S
    
    X_s = S * X
    X_r = R * X
    
    x_s = istft(X_s, hop_length, n_fft)
    x_r = istft(X_r, hop_length, n_fft)
    
    # Stage 2: Transients vs Noise
    n_fft_2 = 512 
    hop_length_2 = 128 
    
    X_r = stft(x_r, n_fft_2, hop_length_2, n_fft_2)
    
    L_h_2 = max(3, int(round(0.2 * sr / hop_length_2)))
    L_v_2 = max(3, int(round(500 * n_fft_2 / sr)))
    
    X_abs_2 = np.abs(X_r)
    X_h_2 = medfilt_horizontal(X_abs_2, L_h_2)
    X_v_2 = medfilt_vertical(X_abs_2, L_v_2)
    
    R_t = X_v_2 / (X_h_2 + X_v_2 + 1e-8)
    
    T = compute_masks(R_t, beta_u_2, beta_l_2)
    N = 1 - T
    
    X_t = T * X_r
    X_n = N * X_r
    
    x_t = istft(X_t, hop_length_2, n_fft_2)
    x_n = istft(X_n, hop_length_2, n_fft_2)
    
    # Ensure all outputs have the same length
    min_length = min(len(x_s), len(x_t), len(x_n))
    x_s = x_s[:min_length]
    x_t = x_t[:min_length]
    x_n = x_n[:min_length]
    
    return x_s, x_t, x_n