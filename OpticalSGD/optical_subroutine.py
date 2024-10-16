import os
from pathlib import Path
import sys
import numpy as np
import torch
import cv2

from OpticalSGD.utils import load_imaging_results
from SLPipeline import blender_process as bp
from SLPipeline.utils import normalize_image
from SLPipeline.gen_pattern import PatternGenerator

class ImagingFunction(torch.autograd.Function):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward_image : torch.Tensor = None      # (h_img, w_img, k)
    backward_jacobian : torch.Tensor = None  # (h_img, w_img, w_pat, k)
    @staticmethod
    def forward(ctx, input:torch.Tensor, index):
        # 似乎让结果与Input无关，也会记录上这个节点！
        ctx.save_for_backward(input, index)
        return ImagingFunction.forward_image[index, :, :]# + (input * 1e-8).sum()

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output的shape是什么?
        # 是 $\partial L / \partial output of ImagingFunction$ 的结果，shape和 ImagingFunction.forward 的输出相同。是(h_selected, w_img, k)
        # Image Jacobian 的shape是 (h_img, w_img, w_pat, k)
        # 这里需要计算 $\partial L / \partial projected_codemat$, shape是 (k, w_pat)
        proj_img, index = ctx.saved_tensors
        w_cam = grad_output.shape[1]
        J_selected = ImagingFunction.backward_jacobian# [index, :, :, :]  # (h_selected, w_img, w_pat, k)
        output = torch.einsum("hmk, hmnk -> hnk", grad_output, J_selected) # / w_cam # (h_selected, w_pat, k)  需要求平均吗？好像不需
        output = torch.mean(output, dim=0).permute(1, 0)    # (k, w_pat)
        print(output.shape)
        return output, None


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
        # output_path = render_proc.get_output_path()
        # fnames = sorted(os.listdir(output_path))
        # imgs = []
        # for fname in fnames:
        #     if not fname.startswith("image"):
        #         continue
        #     p = os.path.join(output_path, fname)
        #     img = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # (h, w)
        #     if img.dtype == np.uint8:
        #         img = normalize_image(img.astype(np.float32), 8)
        #     imgs.append(torch.from_numpy(img))
        # ImagingFunction.forward_image = torch.stack(imgs, dim=-1)
        ImagingFunction.forward_image = torch.from_numpy(render_proc.load_rendered_images()).to(ImagingFunction.device)
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
        J:torch.Tensor = torch.zeros((h_img, w_img, w_pat, k))  # (将h_img当成batch_size)
        blk_num = w_pat / B
        blk_rest = w_pat % B
        with torch.no_grad():
            for i, a, ind in ImagingFunction.jacobian_schedule(B, w_pat, k):
                pat = pattern + a*h
                pats = pattern_saver.codematrix2patterns(pat.detach().cpu().numpy())
                pattern_saver.save_all_to_dir(pats, pat)
                img = ImagingFunction.imaging(render_proc)
                d_a = (img - ori_image) / h   # (h_img, w_img, k)
                # 这样的d_a算出了几列（l1, l2, ... lt）的J之和，假设这些列的每行里只有一个不为0...
                if i < blk_rest:
                    n_blks = blk_num + 1
                else:
                    n_blks = blk_num
                # n_blks == ind.shape[-1]
                # ind = ind.unsqueeze(dim=0).repeat(w_img, 1)  # (w_img, n_blks), d_a 的第w_img列是J的ind[w_img, :]列之和
                # for h in range(h_img):
                #     choice = torch.randint(0, n_blks, size=(w_img))
                #     J[torch.ones((w_img)) * h, torch.arange(w_img), choice] = d_a[h]
                ind = ind.unsqueeze(dim=0).unsqueeze(dim=0).repeat(h_img, w_img, 1)  # (h_img, w_img, n_blks)
                choice = torch.gather(ind, -1, torch.randint(0, n_blks, (h_img, w_img, 1))).unsqueeze(-1).repeat(1, 1, 1, k)  # (h_img, w_img, 1, k)
                J.scatter_(-2, choice, d_a)
                # J[torch.arange(h_img).unsqueeze(-1).repeat(1, w_img), torch.arange(w_img).unsqueeze(0).repeat(h_img, 1), choice] = d_a
        ImagingFunction.backward_jacobian = J


    @staticmethod
    def jacobian_schedule(B, N, K):
        '''
        B: the number of direction vectors a \\
        N: the width of a pattern   \\
        K: the number of patterns  \\
        h: a small value
        '''
        r = N % B
        for i in range(B):
            a = torch.zeros((K, N))
            ind = torch.arange(i, N, B)
            a[:, ind] = 1
            yield i, a, ind