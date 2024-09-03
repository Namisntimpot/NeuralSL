import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from SLPipeline.utils import ZNCC, ZNCC_torch

def expand_p_neighbor(a:np.ndarray, b:np.ndarray, p:int):
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

    for i in range(-t, 0):
        a_tmp = torch.zeros_like(a)
        b_tmp = torch.zeros_like(b)
        a_tmp[..., -i:, :] = a[..., :w_cam+i, :]
        b_tmp[:,-i:] = b[:, :w_pat+i]
        a_tmp[..., :-i, :] = a[...,0:1,:]
        b_tmp[:, :-i] = b[...,0:1]
        a_arr.append(a_tmp)
        b_arr.append(b_tmp)
    a_arr.append(a)
    b_arr.append(b)
    for i in range(1, t+1):
        a_tmp = torch.zeros_like(a)
        b_tmp = torch.zeros_like(b)
        a_tmp[..., :w_cam-i, :] = a[..., i:, :]
        b_tmp[:,:w_pat-i] = b[:, i:]
        a_tmp[..., w_cam-i:, :] = a[...,-1:,:]
        b_tmp[:, w_pat-i:] = b[...,-1:]
        a_arr.append(a_tmp)
        b_arr.append(b_tmp)
    # for i in range(-t, t+1):
    #     a_tmp = torch.zeros_like(a)
    #     b_tmp = torch.zeros_like(b)
    #     a_tmp[..., -i : w_cam - i, :] = a[..., i : w_cam+i, :]
    #     b_tmp[:, -i : w_pat - i] = b[:, i : w_pat + i]
    #     if i < 0:
    #         a_tmp[..., :-i, :] = a[..., 0, :]
    #         b_tmp[:, :-i] = b[:, 0]
    #     elif i>0:
    #         a_tmp[..., w_cam - i:, :] = a[..., -1, :]
    #         b_tmp[:, w_pat-i:] = b[:, -1]
    #     a_arr.append(a_tmp)
    #     b_arr.append(b_tmp)
    ap = torch.cat(a_arr, dim=-1)
    bp = torch.cat(b_arr, dim=-2)
    return ap, bp


def ZNCC_p(a:torch.Tensor, b:torch.Tensor, p):
    '''
    a: (..., w_cam, k), camera's code array
    b: (k, w_pat), projector's code matrix
    p: p-neighbor, perferably an odd number
    '''
    ap, bp = expand_p_neighbor(a, b, p)
    return ZNCC_torch(ap, bp)
    

class ZNCC_NN(nn.Module):
    def __init__(self, p, n_pat, n_layers = 2) -> None:
        super().__init__()
        self.p = p
        self.n_pat = n_pat
        n_feat = p * n_pat
        self.F_proj = nn.Sequential(
            *[Block(n_feat, n_feat) for _ in range(n_layers)]
        )
        self.F_cam = nn.Sequential(
            *[Block(n_feat, n_feat) for _ in range(n_layers)]
        )
    
    def forward(self, cam_code:torch.Tensor, proj_code:torch.Tensor, argmax=False):
        '''
        cam_code: (h_img, w_img, k), camera's code array
        proj_code: (k, w_pat), what the projector projects
        '''
        p_cam_code, p_g_proj_code = expand_p_neighbor(cam_code, proj_code, self.p)  # (h_img, w_img, k*p), (k*p, w_pat)
        F_p_cam_code = self.F_cam(p_cam_code)  # (h_img, w_img, k*p)
        F_p_g_proj_code = self.F_proj(p_g_proj_code.T).T  # (k*p, w_pat)
        zncc = ZNCC_torch(F_p_cam_code + p_cam_code, F_p_g_proj_code + p_g_proj_code)  # (h_img, w_img, w_pat)
        if argmax:
            return torch.max(zncc, dim=-1)[1]  # (h_img, w_img), 返回与哪个列匹配成功.
        return zncc
        


class Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear0 = nn.Linear(in_dim, out_dim)
        self.linear1 = nn.Linear(out_dim, in_dim)

    def forward(self, x):
        xp = self.linear1(F.relu(self.linear0(x)))
        return x + xp