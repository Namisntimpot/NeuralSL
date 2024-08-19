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
        
        return torch.stack(imgs, dim=-1)
    
    @staticmethod
    def update_image_jacobian(
        render_proc:bp.BlenderSubprocess,
        
    ):
        pass

    @staticmethod
    def jacobian_schedule(B:int, w:int):
        pass