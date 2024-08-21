import os
import numpy as np
import cv2

from .utils import ZNCC

class SLPipeline:
    def __init__(self, Kp, Kc, R, T, code_matrix) -> None:
        '''
        instantiate and initialize a projector-camera system.  
        Kp: projector's intrinsic matrix  
        Kc: camera's intrinsic matrix
        R:  the rotation matrix of the camera relative to the projector, (3, 3)
        T:  the translation vector of the camera relative to the projector, (3) (in mm)
        code_matrix: the code matrix. (n_patterns, width)
        '''
        self.Kp, self.Kc = Kp, Kc
        self.R, self.T, self.code_matrix = R, T, code_matrix

        T3x1 = np.expand_dims(T, axis=-1)
        trans3x4 = np.concatenate([R, T3x1], axis=-1)

        self.Mp = np.matmul(self.Kp, trans3x4)
        self.Kc_inv = np.linalg.inv(self.Kc)

    def images_to_code_array(self, images):
        '''
        turn a sequence of images into the code arrays. the images must be sorted according to the patterns' order
        return: (h, w, n_patterns)
        '''
        assert len(images) == self.code_matrix.shape[0]
        return np.stack(images, axis=-1)
    
    def match(self, image_code_array:np.ndarray, sim_func = ZNCC):
        '''
        match the code array of each pixel with the columns in the projector's code matrix.
        this function use the default ZNCC decorder, i.e. it finds the column that maximize the ZNCC value.
        return: (h, w), the matched column's index of each pixel
        '''
        zncc = sim_func(image_code_array, self.code_matrix)
        return np.argmax(zncc, axis=-1)
    
    def get_depth(self, matched_indices:np.ndarray):
        h, w = matched_indices.shape[-2], matched_indices.shape[-1]
        mp1 = np.expand_dims(self.Mp[...,0,:3], axis=0)
        mp14 = self.Mp[...,0,3]
        
        y, x = np.mgrid[:h, :w]
        z = np.ones((h, w, 1))
        pixel_coords = np.expand_dims(np.concatenate([np.expand_dims(x, axis=-1), np.expand_dims(y, axis=-1), z], axis=-1), axis=-1)   # h, w, 3, 1

        z = mp14 / (matched_indices - np.squeeze(np.einsum('hi, ij, ...jk -> ...hk', mp1, self.Kc_inv, pixel_coords)))
        return z