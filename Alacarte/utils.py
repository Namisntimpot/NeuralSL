import os
import numpy as np
import torch
from numpy import pi


def build_frequence_clip_matrix(n_pat, freq_lower, freq_upper):
    '''
    from WZ  
    注意，这个矩阵是给n_pat在最后一个维度的code matrix准备的.
    '''
    xnp = np.arange(n_pat, dtype=np.float32)
    xnp = -xnp
    xnp = 2.0 * pi * xnp / n_pat
    XXnp = np.tile(xnp.reshape(1, n_pat), [n_pat, 1])
    XXnp = XXnp * np.arange(n_pat, dtype=np.float32).reshape(n_pat, 1)
    # [[2pi 1*1 /n, 2pi 1*2 / n, 2pi 1*3 / n, 2pi 1*4 / n, 2pi 1*5 / n...],
    #  [2pi 2*1 /n, 2pi 2*2 / n...]], XXnp_{i,j} = 2pi * i*j  / n
    # K: 能够被保留的fft参数；X: n_pat
    XXcosnp = np.cos(XXnp)
    XXsinnp = np.sin(XXnp)
    #
    # assert(freq_upper > 0)
    if freq_upper <= 0 or freq_upper >= n_pat // 2:
        return np.eye(n_pat, dtype=np.float32)
    #
    frearray = np.zeros(shape=(n_pat, ), dtype=bool)
    frearray[0] = True
    frearray[1:freq_upper + 1] = True
    frearray[-freq_upper:] = True
    #
    if (freq_lower > 0):
        assert freq_upper > freq_lower
        frearray[1:freq_lower + 1] = False
        frearray[-freq_lower:] = False
    #
    KXcosnp = XXcosnp[frearray, :]  # 取出这个频率范围内的cos, sin  # (F, n), F是要保留的, n是n列...
    KXsinnp = XXsinnp[frearray, :]
    XKcosnp = np.transpose(KXcosnp) # (n, F)
    XKsinnp = np.transpose(KXsinnp)
    # ret_{i,j} = sum_{f} XKcosnp_{i,f}*KXcosnp_{f, j}
    return (np.matmul(XKcosnp, KXcosnp) + np.matmul(XKsinnp, KXsinnp)) / n_pat  # (n, n)


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
    n_pat = 800
    random_phases = np.exp(2j * np.pi * np.random.rand(n_pat // 2 - 1))
    freq = np.zeros((n_pat), dtype=np.complex64)
    freq[1:n_pat//2] = random_phases
    freq[n_pat//2+1:] = np.conj(random_phases)
    freq[0] = freq[n_pat//2] = 0

    freq[17:] = 0
    
    t = np.fft.ifft(freq).real
    t = (t - t.min()) / (t.max() - t.min())

    x = np.arange(n_pat)
    plt.plot(x, t)
    plt.show()
    print(t)