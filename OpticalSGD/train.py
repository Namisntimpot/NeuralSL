import json
import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from OpticalSGD.optical_subroutine import ImagingFunction
from OpticalSGD.decoder import ZNCC_NN
from OpticalSGD.pattern import OpticalSGDPattern
from OpticalSGD.utils import resolve_path
from SLPipeline.blender_process import BlenderSubprocess
from SLPipeline.utils import compute_coresponding_from_depth

def parse_args():
    parser = argparse.ArgumentParser(description='Optical SGD training')
    parser.add_argument('--args', type=str, default="OpticalSGD/args.json")
    args = parser.parse_known_args()
    args_path = args.args
    with open(args_path, 'r') as f:
        args = json.load(f)
    return args


class TrainOpticalSGDPattern:
    def __init__(self, args):
        self.args = args
        self.h_cam, self.w_cam = args['h_cam'], args['w_cam']
        self.h_pat, self.w_pat = args['h_pat'], args['w_pat']
        self.n_pat = args['n_pat']
        self.maxF = args['maxF']
        self.tau = args['softmax_tau']

        self.iters_per_Jimg = args['Jimg_update']
        self.B = args['Jimg_B']
        self.h = args['Jimg_h']
        self.n_iters = args['n_iters']
        self.device = torch.device(args['device'])

        self.output_base_dir = Path(args['output_base_dir'])
        self.output_pattern_dir = resolve_path(Path(args['output_pattern_dir']), self.output_base_dir)
        self.output_log_dir = resolve_path(Path(args['output_log_dir']), self.output_base_dir)

        self.blender_subprocess = BlenderSubprocess(
            exe_path=args['blender']['blender_exe_path'],
            scene_path=args['blender']['blender_scene_path'],
            pattern_path=args['blender']['blender_pattern_path'],
            output_path=args['blender']['blender_output_path'],
            script_path=args['blender']['blender_script_path'],
            cwd=args['blender']['blender_cwd'],
        )

        self.pattern = OpticalSGDPattern(
            self.w_pat, self.h_pat, self.n_pat, self.maxF, args['proj_n_g'], str(self.output_pattern_dir)
        ).to(self.device)
        self.decoder = ZNCC_NN(args['decoder_p'], self.n_pat, args['decoder_n_layers']).to(self.device)

        self.optimizer = optim.RMSprop(
            [
                {'params': self.pattern.parameters()},
                {'params': self.decoder.patameters()}
            ],
            lr=args['lr'],
        )
        self.optim_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args['lr_decay'][1], gamma=args['lr_decay'][0])

    def loss_func(self, gt_matched_indices:torch.Tensor, zncc:torch.Tensor):
        '''
        gt_matched_indices: (h_img, w_img)
        zncc: (h_img, w_img, w_pat)
        '''
        batchsize = zncc.shape[-1]
        sf = F.softmax(self.tau * zncc.to(torch.float64), dim=-1)  # (h_img, w_img, w_pat)
        index = torch.arange(self.w_pat).unsqueeze(0).unsqueeze(0).repeat(self.h_img, self.w_img, 1).to(self.device) # (h_img, w_img, w_pat)
        err = torch.abs(index - gt_matched_indices.unsqueeze(-1))  # (h_img, w_img, w_pat)
        err = torch.einsum('ijk, ijk -> ij', sf, err).to(gt_matched_indices.dtype)
        return err.sum() / batchsize

    def trainloop(self):
        pass