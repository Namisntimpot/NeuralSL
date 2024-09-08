import numpy as np
import torch

from SLPipeline.utils import *
from SLPipeline.hardware_settings import HardwareSettings

def get_local_ray_directions(W, H, fx, fy, cx, cy, use_pixel_centers=True, OPENGL = False):
    pixel_center = 0.5 if use_pixel_centers else 0
    i, j = np.meshgrid(
        np.arange(W, dtype=np.float32) + pixel_center,
        np.arange(H, dtype=np.float32) + pixel_center,
        indexing='xy'
    )
    i, j = torch.from_numpy(i), torch.from_numpy(j)

    if OPENGL:
        directions = torch.stack([(i - cx) / fx, -(j - cy) / fy, -torch.ones_like(i)], -1) # (H, W, 3)

    else:
        directions = torch.stack([(i - cx) / fx, (j - cy) / fy, torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_world_rays(directions:torch.Tensor, c2w:torch.Tensor, keepdim=False):
    '''
    directions: (..., 3)
    c2w: (4,4)
    return: rays_o (..., 3), rays_d (..., 3)
    '''
    assert directions.shape[-1] == 3

    R = c2w[:3, :3]
    T = c2w[:3, 3]

    w_directions = torch.einsum("...k, jk -> ...j", directions, R)
    w_origins = T.expand(w_directions.shape)

    if not keepdim:
        w_directions = w_directions.reshape(-1, 3)
        w_origins = w_origins.reshape(-1, 3)
    return w_origins, w_directions


def compute_rays_intersection(r1_o:torch.Tensor, r1_d:torch.Tensor, r2_o:torch.Tensor, r2_d:torch.Tensor):
    '''
    Assuming two straight lines are collinear and we do not check it here.
    return t1, t2 (distance)
    '''
    delta = r2_o - r1_o
    t3 = delta.norm(dim=-1)
    cos1 = (r1_d * delta).sum(dim=-1) / (r1_d.norm(dim=-1) * t3)
    cos2 = (r2_d * (-delta)).sum(dim=-1) / (r2_d.norm(dim=-1) * t3)
    sin1 = torch.sqrt(1 - cos1**2)
    sin2 = torch.sqrt(1 - cos2**2)
    cos3 = -(cos1 * cos2 - sin1 * sin2)

    k = sin2 / sin1
    t2 = torch.sqrt(t3**2 / (k**2 + 1 - 2*k*cos3))
    k = sin1 / sin2
    t1 = torch.sqrt(t3**2 / (k**2 + 1 - 2*k*cos3))
    return t1, t2