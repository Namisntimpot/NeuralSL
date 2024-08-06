import os
import numpy as np
import torch
from torch import nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from SLPipeline.gen_pattern import PatternGenerator
from SLPipeline.utils import ZNCC_torch
from .utils import *


class GeometricConstraint:
    def __init__(self, pattern_w, camera_w, Kp, Kc, R, T, near_depth, far_depth, left, right) -> None:
        '''
        assuming the bounding box is parallel to the camera plane, left and right are relative to the camera origin.
        in mm. 
        for R, T, only identity R and translation along x-axis are considered
        '''
        self.camera_w, self.pattern_w = camera_w, pattern_w
        self.Kp, self.Kc, self.R, self.T = Kp, Kc, R, T
        self.near, self.far, self.left, self.right = near_depth, far_depth, left, right
        self.trans = np.concatenate([R, np.expand_dims(T, axis=-1)], axis=-1)
        self.Mp = np.matmul(self.Kp, self.trans)
        self.G = None
        self.wp_bound_z, self.wp_bound_hori, self.wc_bound_hori = None, None, None
        self.G = self.get_geom_constraint_matrix()
        pass

    def get_geom_constraint_matrix(self):
        if self.G is None:
            cam_width = np.expand_dims(np.arange(self.camera_w), axis=-1)  # (w, 1)
            cam_width = np.concatenate([cam_width, cam_width], axis=-1)    # (w, 2)
            z_bound = np.array([self.far, self.near])                      # (   2)
            mp11, mp13, mp14, kc11, kc13 = self.Mp[0,0], self.Mp[0,2], self.Mp[0,3], self.Kc[0,0], self.Kc[0,2]
            wp_bound_z = mp11 * (cam_width - kc13) / kc11 + mp13 + mp14 / z_bound  # (w, 2), 0: lower, 1: upper
            wp_bound_z = np.clip(wp_bound_z, cam_width, self.pattern_w-1)
            if self.left is None or self.right is None:
                wp_bound_horiz = np.array([0, self.pattern_w-1])
                wc_bound_horiz = np.array([0, self.camera_w -1])
            else:
                corners = np.array(
                    [[self.left, 0, self.near],
                     [self.left, 0, self.far],
                     [self.right, 0, self.near],
                     [self.right, 0, self.far]]
                ).T
                pix_cam = np.matmul(self.Kc, corners) / corners[2, :]
                pix_proj = np.matmul(self.Mp, np.concatenate([corners, np.ones((1, 4))], axis=-2)) / corners[2, :]
                pix_cam_w, pix_proj_w = np.clip(pix_cam[0, :], 0, self.camera_w-1), np.clip(pix_proj[0, :], 0, self.pattern_w-1)
                wp_bound_horiz = np.array([np.min(pix_proj_w[0], pix_proj_w[1]), np.max(pix_proj_w[2], pix_proj_w[3])])
                wc_bound_horiz = np.array([np.min(pix_cam_w[0], pix_cam_w[1]), np.max(pix_cam_w[2], pix_cam_w[3])])
            
            G_horiz = np.zeros((self.pattern_w, self.camera_w))
            G_horiz[wp_bound_horiz[0]:wp_bound_horiz[1]+1, wc_bound_horiz[0]:wc_bound_horiz[1]+1] = 1
            G_z = np.zeros((self.pattern_w, self.camera_w))
            for q in range(self.camera_w):
                G_z[wp_bound_z[q, 0]:wp_bound_z[q, 1]+1, q] = 1
            self.G = G_z * G_horiz
            self.wp_bound_z = wp_bound_z
            self.wp_bound_hori = wp_bound_horiz
            self.wc_bound_hori = wc_bound_horiz
        return self.G
    

class AlacartePattern(PatternGenerator):
    def __init__(
            self, pat_width, pat_height, n_patterns, 
            cam_width, tolerance, geom_constraints:GeometricConstraint, ambient_max = 0.5, mu=300,
            output_dir=None, format='png', defocus = False, rho:int = None, maxF = None,
            device:str = 'cuda'
        ) -> None:
        super().__init__(pat_width, pat_height, n_patterns, output_dir, format)
        self.cam_w = cam_width
        self.tolerance = tolerance
        self.G = geom_constraints
        self.ambient_max = ambient_max
        self.mu = mu
        self.maxF = maxF
        self.defocus = defocus
        self.rho = rho        
        
        self.device = torch.device(device)
        if self.defocus:
            self.defocus_matrix = self.get_simple_defocus_kernel()
        self.code_matrix = nn.Parameter(torch.rand((self.n_patterns, self.width)).to(self.device))  # (n_patterns, pat_width)
        self.optimizer = optim.Adam([self.code_matrix], 0.01)

        
    def transport(self, transport_matrix:torch.Tensor, ambient:torch.Tensor, noise:torch.Tensor):
        if self.defocus:
            C = torch.matmul(self.code_matrix, self.defocus_matrix)
        else:
            C = self.code_matrix
        ret = torch.matmul(C, transport_matrix) + ambient + noise
        return torch.transpose(ret, dim0=-2, dim1=-1)  # place the code at the last axis.
    
    def sample_conditions(self, n_samples):
        '''
        return: matched_indices, direct-only T, ambient, noise
        '''
        noise = torch.normal(0, 0.01, size=(n_samples, self.n_patterns, self.cam_w))
        # noise = torch.randn((n_samples, self.n_patterns, self.cam_w))
        ambient = torch.rand((n_samples, self.cam_w)) * self.ambient_max
        ambient = torch.unsqueeze(ambient, dim=1)
        ambient = ambient.repeat(1, self.n_patterns, 1)
        # randomly generate direct-only T. first assign random stereo matched columns, which specifies the location of the only non-zero element in each column of T
        # then assign random values to those element.
        if self.G is None:
            off = torch.arange(self.cam_w)
            scale= (self.width - off) / self.width
            matched_indices = torch.clip(torch.round(self.width * (torch.rand((n_samples, self.cam_w))) * scale + off), off, torch.tensor(self.width-1)).long()
            T = torch.zeros((n_samples, self.width, self.cam_w))
            T[:, matched_indices, off] = torch.rand((n_samples, self.cam_w))
        else:
            raise NotImplementedError
        return matched_indices.to(self.device), T.to(self.device), ambient.to(self.device), noise.to(self.device)

    def compute_error(self, observation:torch.Tensor, matched_indices:torch.Tensor):
        n_samples = observation.shape[0]
        # TODO: 只选取matched_indices的部分参与计算，从而实现算巨大的mu.
        if self.defocus:
            exp_zncc = torch.exp(self.mu * (ZNCC_torch(observation, torch.matmul(self.code_matrix, self.defocus_matrix))))
        else:
            exp_zncc = torch.exp(self.mu * (ZNCC_torch(observation, self.code_matrix)))   # (n_samples, cam_w, pat_w)
        norm = exp_zncc / (torch.sum(exp_zncc, dim=-1, keepdim=True))
        aux_mat = torch.zeros((self.width, self.cam_w)).to(self.device)
        for i in range(-self.tolerance, self.tolerance + 1):
            aux_mat[..., torch.clip(matched_indices + i, 0, self.width-1), torch.arange(self.cam_w)] = 1
        correct = torch.einsum('...ij, ji -> ...i', norm, aux_mat)
        correct = torch.sum(correct, dim=-1)
        expected_error = torch.sum(self.cam_w - correct) / n_samples
        return expected_error
    
    def get_simple_defocus_kernel(self):
        '''
        return the simple defocus kernel defined in the supp, which is independent of the x-axis of the image.
        so it's a single matrix pattern_w * pattern_w.
        '''
        defocus_matrix = np.zeros((self.width, self.width))
        for i in range(self.width):
            defocus_matrix[max(0, i - self.rho) : min(self.width, i + self.rho), i] = 1. / (2 * self.rho)
        return torch.from_numpy(defocus_matrix).to(self.device)

    def optimize(self, samples_per_iter = 2, n_iters=300, logdir: str = None):
        # sample 500 conditions for evaluation.
        self.eval_conditions = self.sample_conditions(500)
        eval_time = 50   # evaluate per ${eval_time} iters

        if logdir is not None:
            global_step = 0
            writer = SummaryWriter(logdir)
        
        for i in range(n_iters):
            samples_matched_indices, samples_T, samples_ambient, samples_noise = self.sample_conditions(samples_per_iter)
            # (n_samples, cam_w), (n_samples, pat_w, cam_w), (n_samples, cam_w), (n_samples, n_patterns, cam_w)
            observations = self.transport(samples_T, samples_ambient, samples_noise)
            error = self.compute_error(observations, samples_matched_indices)
            self.optimizer.zero_grad()
            error.backward()
            self.optimizer.step()

            if i % eval_time == 0:
                eval_err = self.evaluate()
                print("[iter %04d]: the evaluation error: %f" % (i, eval_err.detach().cpu().item()))
                if logdir is not None:
                    writer.add_scalar('error in eval', eval_err.detach().cpu().item(), global_step)

            if logdir is not None:
                writer.add_scalar('error per iter', error.detach().cpu().item(), global_step)
                global_step += 1

        print("Iterations finished.")


    def pattern_post_process(self):
        '''
        limit the range of the pattern pixels to [0, 1]
        (unimplemented) limit the frequence of the pattern to [0, maxF]
        '''
        with torch.no_grad():
            torch.clip_(self.code_matrix, 0, 1)
            if self.maxF is not None:
                self.code_matrix = clip_frequencies(self.code_matrix, self.maxF)
                torch.clip_(self.code_matrix, 0, 1)

    def evaluate(self):
        '''
        the "optimize" method must be called earlier.
        '''
        with torch.no_grad():
            eval_matched, eval_T, eval_ambient, eval_noise = self.eval_conditions
            obs = self.transport(eval_T, eval_ambient, eval_noise)
            err = self.compute_error(obs, eval_matched)
            return err
        
    def save(self, pt_path = None):
        if pt_path is not None:
            torch.save(self.code_matrix, pt_path)
        code = self.code_matrix.detach().cpu()  # (n_patterns, pattern width)
        patterns = code.unsqueeze(dim=1)
        patterns: np.ndarray = patterns.repeat(1, self.height, 1).numpy()
        self.save_all_to_dir(patterns, code.numpy())
    