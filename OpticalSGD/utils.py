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
    

def load_imaging_results(path:str, need_gt = False):
    '''
    load the imaging results. For now, 'imaging results' are the Blender's rendering results.  
    images' filenames are started with 'image', while depth's filename is started with 'depth'.  
    return: image_code_arrays if not need_gt else (image_code_arrays, gt_depth)
    '''
    img_paths = sorted(glob.glob(os.path.join(path, 'image*')))
    imgdata = []
    for imgp in  img_paths:
        img = normalize_image(cv2.imread(imgp, cv2.IMREAD_UNCHANGED), bit_depth=8)
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