import os
import numpy as np
from pathlib import Path
import cv2
import torch



def resolve_path(path:Path, basepath:Path = None):
    if path.is_absolute():
        return path.as_posix()
    else:
        if basepath is None:
            basepath = Path.cwd()
        return basepath.joinpath(path).resolve().as_posix()
    

def load_imaging_results(ret_path:str, need_gt = False):
    '''
    load the imaging results. For now, 'imaging results' are the Blender's rendering results.  
    images' filenames are started with 'image', while depth's filename is started with 'depth'.  
    return: image_code_arrays if not need_gt else (image_code_arrays, gt_depth, gt_coresponding)
    '''
