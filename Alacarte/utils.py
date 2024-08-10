import os
import numpy as np
import torch

def clip_frequencies(t:torch.Tensor, max_f):
    n_patterns, w = t.shape
    if w <= max_f:
        return t
    freq_domain = torch.fft.fft(t, dim=1)
    freq_domain[:, max_f+1 : ] = 0
    ifft = torch.fft.ifft(freq_domain, dim=1)
    return ifft.real

def freq_domain_random_init(height, width, maxF = None):
    random_phases = torch.exp(2j * torch.pi * torch.rand((height, width // 2 - 1)))
    freq = torch.zeros((height, width), dtype=torch.complex128)
    freq[:, 1:width//2] = random_phases
    freq[:, width//2+1:] = torch.conj(random_phases)
    freq[:, 0] = freq[:, width // 2] = 0
    if maxF is not None:
        freq[:, maxF+1:] = 0
    arr = torch.fft.ifft(freq).real
    rows_min = arr.min(dim=-1)[0].unsqueeze(dim=-1)
    rows_max = arr.max(dim=-1)[0].unsqueeze(dim=-1)
    # to [0, 1]
    arr = (arr - rows_min) / (rows_max - rows_min)
    return arr
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 800
    random_phases = np.exp(2j * np.pi * np.random.rand(n // 2 - 1))
    freq = np.zeros((n), dtype=np.complex64)
    freq[1:n//2] = random_phases
    freq[n//2+1:] = np.conj(random_phases)
    freq[0] = freq[n//2] = 0

    freq[17:] = 0
    
    t = np.fft.ifft(freq).real
    t = (t - t.min()) / (t.max() - t.min())

    x = np.arange(n)
    plt.plot(x, t)
    plt.show()
    print(t)