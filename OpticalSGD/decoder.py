import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from SLPipeline.utils import ZNCC, ZNCC_torch

def ZNCC_p(a:torch.Tensor, b:torch.Tensor, p):
    '''
    a: (..., w_cam, k), camera's code array
    b: (k, w_pat), projector's code matrix
    p: p-neighbor, perferably an odd number
    '''
    t = p // 2
    k, w_pat = b.shape[-2], b.shape[-1]
    w_cam = a.shape[-2]
    a_arr = []
    b_arr = []
    with torch.no_grad():
        for i in range(-t, t+1):
            a_tmp = torch.zeros_like(a)
            b_tmp = torch.zeros_like(b)
            a_tmp[..., -i : w_cam - i, :] = a[..., i : w_cam+i, :]
            b_tmp[:, -i : w_pat - i] = b[:, i : w_pat + i]
            if i < 0:
                a_tmp[..., :-i, :] = a[..., 0, :]
                b_tmp[:, :-i] = b[:, 0]
            elif i>0:
                a_tmp[..., w_cam - i:, :] = a[..., -1, :]
                b_tmp[:, w_pat-i:] = b[:, -1]
            a_arr.append(a_tmp)
            b_arr.append(b_tmp)
        ap = torch.cat(a_arr, dim=-1)
        bp = torch.cat(b_arr, dim=-2)
    return ZNCC_torch(ap, bp)
    

class ZNCC_NN(nn.Module):
    def __init__(self, p, n_pat, n_layers = 2, n_g_segments = 32) -> None:
        super().__init__()
        self.p = p
        self.n_pat = n_pat
        self.n_g_segs = n_g_segments
        self.param_g = nn.Parameter(torch.arange(0, n_g_segments+1) / n_g_segments)  # g函数的参数是前序和
        n_feat = p * n_pat
        self.F_proj = nn.Sequential(
            *[Block(n_feat, n_feat) for _ in range(n_layers)]
        )
        self.F_cam = nn.Sequential(
            *[Block(n_feat, n_feat) for _ in range(n_layers)]
        )

    def g_function(self, x:torch.Tensor):
        x_coords = torch.arange(0, self.n_g_segs) / self.n_g_segs
        # 先找到那个恰好比x小的节点
        with torch.no_grad():
            delta = torch.abs(x_coords - x.unsqueeze(-1))
            _, ind = delta.min(dim=-1)  # ind的shape应该是(h_cam, w_cam, k)
        aux = torch.zeros_like(x.unsqueeze(-1).repeat_interleave(self.n_g_segs, -1))
        
        


class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear0 = nn.Linear(in_dim, out_dim)
        self.linear1 = nn.Linear(out_dim, in_dim)

    def forward(self, x):
        xp = self.linear1(F.relu(self.linear0(x)))
        return x + xp