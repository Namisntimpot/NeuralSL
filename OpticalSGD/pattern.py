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
        self.g_param = nn.Parameter(torch.arange(0, n_g_segs+1) / n_g_segs)  # 初始化为{0, 1/32, 2/32,..., 31/32, 1}
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
        seg_indices = (x * self.n_g_segs).floor().long()  # 先把x值映射到[0, 1]区间，然后乘以n_g_segs，得到每个像素点对应的segment的index
        y0 = self.g_param[seg_indices]
        y1 = self.g_param[torch.clamp(seg_indices+1, max=self.n_g_segs)]
        x_relative = (x - seg_indices.float() / self.n_g_segs) * self.n_g_segs
        y = y0 + (y1-y0) * x_relative
        return y