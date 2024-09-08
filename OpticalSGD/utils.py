import os
import numpy as np
from pathlib import Path
import cv2
import torch
import glob
from SLPipeline.utils import *


def resolve_path(path:Path, basepath:Path = None):
    if path.is_absolute():
        return path.as_posix()
    else:
        if basepath is None:
            basepath = Path.cwd()
        return basepath.joinpath(path).resolve().as_posix()
    

def load_imaging_results(path:str, need_gt = False, img_prefix = "image"):
    '''
    load the imaging results. For now, 'imaging results' are the Blender's rendering results.  
    images' filenames are started with 'image', while depth's filename is started with 'depth'.  
    return: image_code_arrays if not need_gt else (image_code_arrays, gt_depth)
    '''
    img_paths = sorted(glob.glob(os.path.join(path, img_prefix+"*")))
    imgdata = []
    for imgp in  img_paths:
        img = normalize_image(cv2.imread(imgp, cv2.IMREAD_UNCHANGED), bit_depth=8)
        if len(img.shape) == 3:
            img = img[:,:,0]
        imgdata.append(img)
    img = np.stack(imgdata, axis=-1)

    if need_gt:
        load_cv2_for_exr()
        depth_path = glob.glob(os.path.join(path, 'depth*'))[0]
        gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) * 1000.0  # convert to mm
        if len(gt_depth.shape) == 3:
            gt_depth = gt_depth[:, :, 0]
        return img, gt_depth
    else:
        return img
    
def load_patterns_results(path:str, pat_prefix = "", n_pat_axis = 0):
    pat_paths = sorted(glob.glob(os.path.join(path, pat_prefix+"*")))
    patdata = []
    for patp in pat_paths:
        pat = cv2.imread(patp, cv2.IMREAD_UNCHANGED)
        if len(pat.shape) == 3:
            pat = pat[0,:,0]
        else:
            pat = pat[0,:]
        pat = normalize_image(pat, bit_depth=8)
        patdata.append(pat)
    return np.stack(patdata, axis=n_pat_axis)