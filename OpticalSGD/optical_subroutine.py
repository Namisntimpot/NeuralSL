import os
from pathlib import Path
import sys
import numpy as np
import torch
import cv2

from SLPipeline import blender_process as bp
from SLPipeline.utils import normalize_image
from SLPipeline.gen_pattern import PatternGenerator

class ImagingFunction(torch.autograd.Function):
    forward_image : torch.Tensor = None
    backward_grad : torch.Tensor = None
    @staticmethod
    def forward(ctx, input, filter, bias, padding, stride):
        ctx.save_for_backward(input, filter, bias)

    @staticmethod
    def backward(ctx, grad_output):
        pass

    @staticmethod
    def imaging(render_proc:bp.BlenderSubprocess):
        '''
        render the image sequence using the newest patterns.  \\
        here we assume that the patterns are already stored in the render_proc.pattern_path.  \\
        return the image sequence as a torch.Tensor, in the shape of (h, w, k), where k is the number of patterns.
        '''
        returncode = render_proc.run_and_wait()
        if returncode!= 0:
            raise Exception("Blender subprocess failed with return code {}".format(returncode))
        output_path = render_proc.get_output_path()
        fnames = sorted(os.listdir(output_path))
        imgs = []
        for fname in fnames:
            if not fname.startswith("image"):
                continue
            p = os.path.join(output_path, fname)
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # (h, w)
            if img.dtype == np.uint8:
                img = normalize_image(img.astype(np.float32), 8)
            imgs.append(torch.from_numpy(img))
        ImagingFunction.forward_image = torch.stack(imgs, dim=-1)
        return ImagingFunction.forward_image
    
    @staticmethod
    def update_image_jacobian(
        render_proc:bp.BlenderSubprocess,
        pattern:torch.Tensor, B, h, h_pat
    ):
        k, w_pat = pattern.shape
        h_img, w_img, _ = ImagingFunction.forward_image
        pat_path = render_proc.get_pattern_path()
        pattern_saver = PatternGenerator(w_pat, h_pat, k, pat_path)
        ori_image = ImagingFunction.forward_image  # (h, w, k)
        J = torch.zeros((w_pat, w_img, k))
        with torch.no_grad():
            for a in ImagingFunction.jacobian_schedule(B, w_pat, k):
                pat = pattern + a*h
                pats = pattern_saver.codematrix2patterns(pat.detach().cpu().numpy())
                pattern_saver.save_all_to_dir(pats, pat)
                img = ImagingFunction.imaging(render_proc)
                d_a = (img - ori_image) / h   # (h_img, w_img, k)
                # 这样的d_a算出了几列（l1, l2, ... lt）的J之和，假设这些列的每行里只有一个不为0..?
        pass

    @staticmethod
    def jacobian_schedule(B, N, K):
        '''
        B: the number of direction vectors a \\
        N: the width of a pattern   \\
        K: the number of patterns  \\
        h: a small value
        '''
        stride = N // B
        r = N % B
        for i in range(B):
            a = torch.zeros((K, N))
            ind = torch.arange(i, N, stride)
            a[:, ind] = 1
            yield a