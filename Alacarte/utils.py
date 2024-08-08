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

    # h, w = t.shape
    # freq_domain = torch.fft.fft2(t)
    # freq_domain_shifted = torch.fft.fftshift(freq_domain)
    # ch, cw = h // 2, w // 2
    # rows_ind, cols_ind = torch.arange(h), torch.arange(w)
    # x, y = torch.meshgrid(rows_ind, cols_ind, indexing='ij')   # np.mgrid[:h, :w]
    # x = x.to(t.device)
    # y = y.to(t.device)
    # mask = (x-ch)**2 + (y-cw)**2 > max_f * max_f
    # freq_domain_shifted[mask] = 0
    # freq_filtered = torch.fft.fftshift(freq_domain_shifted)
    # return torch.fft.ifft2(freq_filtered)
    # freq_domain[freq_domain > max_f] = 0
    # return torch.fft.ifft2(freq_domain).real   
    

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