import os
import numpy as np
import torch

def clip_frequencies(t:torch.Tensor, max_f):
    '''
    a low pass filter. a circle region which radius is max_f is reserved
    '''
    n_patterns, w = t.shape
    if w <= max_f:
        return t
    freq_domain = torch.fft.fft(t, dim=1)
    index = torch.arange(max_f+1, w, step=1)
    freq_domain[:, index] = 0
    return torch.fft.ifft(freq_domain, dim=1).real  # must .real! although the imag is always 0

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