if __name__ == '__main__':
    import os
    os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nerfacc import accumulate_along_rays, render_weight_from_alpha

from SLPipeline.hardware_settings import HardwareSettings

from TurboSL.utils import *

class TurboSLPixelwiseDecoder:
    def __init__(self, s, hardware_settings:HardwareSettings, imgh, imgw, patw, npat, device=torch.device("cpu")) -> None:
        self.s = s
        self.hardware = hardware_settings
        self.imgh, self.imgw, self.patw, self.npat = imgh, imgw, patw, npat
        self.device = device
        self.cam_c2w = torch.eye(4, device=device).float()
        # warning: hardsettings units issue here (mm)
        self.proj_c2w = RT2TransformMatrix(hardware_settings.R, hardware_settings.T / 1000, want='4x4')
        self.proj_c2w = torch.from_numpy(np.linalg.inv(self.proj_c2w)).float().to(device)
        
        self.fx_cam, self.cx_cam, self.fy_cam, self.cy_cam = decomposite_instrisic(self.hardware.cam_intri)
        self.fx_proj,self.cx_proj,self.fy_proj,self.cy_proj= decomposite_instrisic(self.hardware.proj_intri)
        self.cam_local_directions = get_local_ray_directions(self.imgw, self.imgh, self.fx_cam, self.fy_cam, self.cx_cam, self.cy_cam, use_pixel_centers=True, OPENGL=False).to(device)
        self.proj_local_directions= get_local_ray_directions(self.patw, self.imgh, self.fx_proj,self.fy_proj,self.cx_proj,self.cy_proj, use_pixel_centers=False, OPENGL=False).to(device)

        self.cam_world_origin, self.cam_world_directions = get_world_rays(self.cam_local_directions, self.cam_c2w, keepdim=True)
        self.proj_world_origin,self.proj_world_directions= get_world_rays(self.proj_local_directions, self.proj_c2w, keepdim=True)
    
    def decode(self, img:torch.Tensor, pat:torch.Tensor):
        '''
        img: (h, w, npat),
        pat: (npat, w) or (w, npat)
        '''
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).to(self.device)
            pat = torch.from_numpy(pat).to(self.device)
        else:
            img = img.to(self.device)
            pat = pat.to(self.device)
        if pat.shape == (self.patw, self.npat):
            pat_ = pat
        else:
            pat_ = pat.permute(1, 0)  # pat_: (w, npat)

        matched_indices = torch.zeros((self.imgh ,self.imgw))

        # for i in tqdm(range(self.imgh)):
        for i in range(300, 301):
            proj_row_rays_o = self.proj_world_origin[i]  # (patw, 3)
            proj_row_rays_d = self.proj_world_directions[i].flip(dims=[0])
            # for j in range(self.imgw):
            for j in range(400, 401):
                cam_ray_o = self.cam_world_origin[i, j]  # (3)
                cam_ray_d = self.cam_world_directions[i, j]  # (3)
                print(proj_row_rays_o[0], proj_row_rays_d[0])
                t_pat_pixels = compute_rays_intersection(proj_row_rays_o, proj_row_rays_d, cam_ray_o, cam_ray_d)[1]  # (patw,)
                print(t_pat_pixels[0])
                import matplotlib.pyplot as plt
                plt.plot(np.arange(self.patw), t_pat_pixels.cpu().numpy())
                t_pat_pixels = t_pat_pixels.unsqueeze(0).repeat(self.patw, 1) # (patw, patw), [i,j]表示如果i是匹配结果，那么j的sdf值大小.
                plt.show()
                sdf = t_pat_pixels[:,:1] - t_pat_pixels  # [i, j]表示如果i是匹配结果，那么j是(强行近似假装的)sdf值的大小
                alpha = self.get_approximate_alpha(sdf) # (patw, patw)
                # ray_indices = torch.arange(self.patw).unsqueeze(-1).repeat(1, self.patw)
                # n_rays = self.patw
                weights, transmission = render_weight_from_alpha(alpha)  # (patw, patw)
                rendered = accumulate_along_rays(weights, pat_.flip([0]).unsqueeze(0).repeat(self.patw, 1, 1))  # (patw, npat)
                imgcode = img[i, j]  # (npat)
                sim = ((rendered - imgcode)**2).sum(dim=-1)  # (patw)
                matched = torch.min(sim, dim=0)[1]
                matched_indices[i, j] = self.patw - matched

        return matched_indices.long()                
                

    def get_approximate_alpha(self, sdf):
        sdf_next = torch.concat([sdf[...,1:], (sdf[...,-1] + sdf[...,-1] - sdf[...,-2])[..., None]], dim=-1)
        prev_cdf = F.sigmoid(sdf * self.s)
        next_cdf = F.sigmoid(sdf_next * self.s)
        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-6) / (c + 1e-6)).clip(0, 1)
        return alpha
    

if __name__ == '__main__':
    hdsetting_path = "data/bust/alacarte-f16-n4/hardware_settings.json"
    hdsettings = HardwareSettings(hdsetting_path)
    decoder = TurboSLPixelwiseDecoder(0.1, hdsettings, 600, 800, 800, 4, torch.device("cuda"))
    
    import os
    import glob
    from OpticalSGD.utils import load_imaging_results, load_patterns_results
    from SLPipeline.utils import visualize_disparity, compute_coresponding_from_depth
    datadir = os.path.dirname(hdsetting_path)
    patdir = "data/patterns/alacarte-f16-n4/"
    imgs, gt_depth = load_imaging_results(datadir, True)          # (imgh, imgw, n_pat)
    gt_coresponding = compute_coresponding_from_depth(gt_depth, decoder.hardware.cam_intri, decoder.hardware.proj_intri, decoder.hardware.R, decoder.hardware.T)
    pats = load_patterns_results(patdir, n_pat_axis=-1)  # (patw, n_pat)
    with torch.no_grad():
        matched = decoder.decode(imgs, pats)
    print(gt_coresponding[300, 400])
    print(matched)