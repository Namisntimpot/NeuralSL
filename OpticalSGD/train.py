import json
import argparse
import os
from pathlib import Path
from enum import Enum
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from OpticalSGD.optical_subroutine import ImagingFunction
from OpticalSGD.decoder import ZNCC_NN
from OpticalSGD.pattern import OpticalSGDPattern
from OpticalSGD.utils import resolve_path

from SLPipeline.blender_process import BlenderSubprocess
from SLPipeline.utils import compute_coresponding_from_depth
from SLPipeline.hardware_settings import HardwareSettings

def parse_args():
    parser = argparse.ArgumentParser(description='Optical SGD training')
    parser.add_argument('--args', type=str, default="OpticalSGD/args.json")
    args = parser.parse_known_args()
    args_path = args.args
    with open(args_path, 'r') as f:
        args = json.load(f)
    return args


class TrainOpticalSGDPattern:
    class mode(Enum):
        SIM = 0
        REAL = 1

    def __init__(self, args):
        self.args = args
        self.__mode = self.mode.SIM if args['mode'] == 'sim' else self.mode.REAL
        
        self.h_cam, self.w_cam = args['h_cam'], args['w_cam']
        self.h_pat, self.w_pat = args['h_pat'], args['w_pat']
        self.n_pat = args['n_pat']
        self.maxF = args['maxF']
        self.tau = args['softmax_tau']

        self.iters_per_Jimg = args['re_Jimg']
        self.iters_per_gt = args['re_gt']   # 多久要重新评估一次gt
        self.B = args['Jimg_B']
        self.h = args['Jimg_h']
        self.n_iters = args['n_iters']
        self.minibatch_rate = args['minibatch_rate']
        self.device = torch.device(args['device'])

        self.hardware_settings = HardwareSettings(args['hardware_config'])

        self.output_base_dir = Path(args['output_base_dir'])
        self.output_pattern_dir = resolve_path(Path(args['output_pattern_dir']), self.output_base_dir)
        self.output_log_dir = resolve_path(Path(args['output_log_dir']), self.output_base_dir)
        self.output_param_dir = resolve_path(Path(args['output_param_dir'], self.output_base_dir))

        self.writer = SummaryWriter(self.output_log_dir)

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

        ImagingFunction.device = self.device

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

    def render_pattern_in_blender_scene(self, need_depth = False):
        self.pattern.gen_pattern(save=True)
        self.blender_subprocess.run_and_wait()
        imgs = torch.from_numpy(self.blender_subprocess.load_rendered_images()).to(self.device)
        if need_depth:
            depth = torch.from_numpy(self.blender_subprocess.load_rendered_depth()).to(self.device)
            return imgs, depth
        return imgs        

    def trainloop(self):
        if self.__mode == self.mode.SIM:
            # 虚拟场景，只用在一开始获取一个 gt
            imgs, gt_depth = self.render_pattern_in_blender_scene(need_depth=True)
            gt_coresponding = compute_coresponding_from_depth(gt_depth, self.hardware_settings.cam_intri, 
                                                              self.hardware_settings.proj_intri, self.hardware_settings.R, self.hardware_settings.T)
            gt_coresponding = torch.from_numpy(gt_coresponding).to(self.device)
        first_iter = True
        for i in range(self.n_iters):
            ######## TODO ############
            # 对pattern沿着横轴做点轮转.
            ##########################
            
            # 分real or sim. real没有实现.
            if self.__mode == self.mode.REAL:
                raise NotImplementedError("Auto-tunning with real hardware and scene is not implemented yet.")

            elif self.__mode == self.mode.SIM:
                # 先应用pattern.
                if first_iter:  # first_iter不会轮转
                    ImagingFunction.forward_image = torch.from_numpy(imgs).to(self.device)
                    first_iter = False
                else:
                    self.pattern.gen_pattern(True, scroll=None)
                    imgs = ImagingFunction.imaging(self.blender_subprocess)
                # 每隔几个iters更新Jimg
                if i % self.iters_per_Jimg == 0:
                    ImagingFunction.update_image_jacobian(self.blender_subprocess, self.pattern.gen_pattern(False, scroll=None), self.B, self.h, self.h_pat)

                # 选取部分行.
                h = imgs.shape[0]
                rand_index = torch.randperm(h)[:int(h * self.minibatch_rate)]
                # 成像
                proj_image = self.pattern.project()
                cam_image = ImagingFunction.apply(proj_image, rand_index)
                # 比较解码
                zncc = self.decoder.forward(cam_image, proj_image)  # (h_selected, w_img, w_pat)
                loss = self.loss_func(gt_coresponding, zncc)

                # 更新梯度
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.optim_scheduler.step()

                # pattern后处理
                self.pattern.regularize_pattern()

                # 记录loss
                self.writer.add_scalar("error", loss.detach().cpu().item(), i)
                if i % 100 == 0:
                    print("[%04d]: %f" % (i, loss.detach().cpu().item()))
            else:
                raise NotImplementedError
        
        # save
        self.save()

    def save(self):
        self.pattern.gen_pattern(True, None)
        torch.save(self.pattern.state_dict(), os.path.join(self.output_param_dir, "pattern.pt"))
        torch.save(self.decoder.state_dict(), os.path.join(self.output_param_dir, "decoder.pt"))
                
