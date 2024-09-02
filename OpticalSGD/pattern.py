import numpy as np
import torch
import torch.nn as nn

from SLPipeline.gen_pattern import PatternGenerator
from Alacarte.utils import clip_frequencies

class OpticalSGDPattern(PatternGenerator, nn.Module):
    def __init__(self, width, height, n_patterns, maxF = None, n_g_segs = 32, output_dir=None, format='png', save_codemat_same_dir=False) -> None:
        super(OpticalSGDPattern, self).__init__(width, height, n_patterns, output_dir, format, save_codemat_same_dir)
        super(PatternGenerator, self).__init__()
        self.n_patterns = n_patterns
        self.width = width
        self.height = height
        self.output_dir = output_dir
        self.format = format
        self.save_codemat_same_dir = save_codemat_same_dir
        self.codemat = nn.Parameter(self.initialize_pattern())
        self.n_g_segs = n_g_segs
        self.g_param = nn.Parameter(torch.arange(0, n_g_segs+1) / n_g_segs)
        self.maxF = maxF

    def initialize_pattern(self):
        return torch.rand((self.n_patterns, self.width)) * 0.1 + 0.45  # (uniform random between 0.45 and 0.55)
    
    def gen_pattern(self, save=True, scroll = None):
        '''
        优化不写在pattern类中，这里直接返回self.codemat，如果需要就保存
        '''
        super().gen_pattern(save)
        if save == True:
            codemat = self.codemat.detach().cpu().numpy()
            self.save_all_to_dir(self.codematrix2patterns(codemat), codemat)
        if scroll is not None:
            ## TODO: 添加论文中的轮转pattern功能!
            raise NotImplementedError
        return self.codemat
    
    def project(self):
        '''
        project the pattern to scenes.
        '''
        return self.g_function(self.codemat)
    
    def regularize_pattern(self):
        with torch.no_grad():
            self.codemat.clip_(0, 1)
            if self.maxF is not None:
                self.codemat.copy_(clip_frequencies(self.codemat, self.maxF))
                self.codemat.clip_(0, 1)

    def g_function(self, x:torch.Tensor):
        x_coords = torch.arange(0, self.n_g_segs, device=x.device) / self.n_g_segs
        # 先找到那个恰好比x小的节点
        with torch.no_grad():
            delta = torch.abs(torch.clip(x_coords - x.unsqueeze(-1), max=0))  # 只取在x value左边的x_coords.
            _, ind = delta.min(dim=-1)  # ind的shape应该是(k, w_pat)
        lower_g_value = self.g_param[ind]
        upper_g_value = self.g_param[ind+1]
        lower_x_value = x_coords[ind]
        g_value = lower_g_value + (upper_g_value - lower_g_value) * self.n_g_segs * (x - lower_x_value)
        return g_value