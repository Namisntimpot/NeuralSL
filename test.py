import os
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import glob
import torch

from SLPipeline.pipeline import SLPipeline
from SLPipeline.hardware_settings import HardwareSettings
from SLPipeline.utils import *

from OpticalSGD.decoder import ZNCC_p_np

base_dir = "Alacarte/testimg/active/"
dbg_dir = os.path.join(base_dir, "debug")
code_matrix_path = os.path.join(base_dir, "code_matrix.png")
img_dir = os.path.join(base_dir, "output")
depth_path = os.path.join(base_dir, "output", "depth0000.exr")
codemat = cv2.imread(code_matrix_path)[:,:,0]
codemat = normalize_image(cv2.imread(code_matrix_path)[:,:,0])


show_vis = True
# remove_albamb = True
p_zncc = 0

# base_dir = "data/"
# scene_obj = "bust"
# pat_family = "alacarte-f16-n4"
# datadir = os.path.join(base_dir, "bust", pat_family)
# patdir = os.path.join(base_dir, "patterns", pat_family)
# img_paths = sorted(glob.glob(os.path.join(base_dir, scene_obj, pat_family, "image*")))
# depth_path = glob.glob(os.path.join(base_dir, scene_obj, pat_family, "depth*"))[0]
# white_path = glob.glob(os.path.join(datadir, "white*"))[0]
# black_path = glob.glob(os.path.join(datadir, "black*"))[0]
# hardware_setting_path = os.path.join(datadir, "hardware_settings.json")

# # load hardware settings
# hdsetting = HardwareSettings(hardware_setting_path)

# load codemat
# codemat = []
# for patp in sorted(glob.glob(os.path.join(patdir, "*"))):
#     im = normalize_image(cv2.imread(patp)[:,:,0])
#     codemat.append(im[0,:])
# codemat = np.stack(codemat, axis=0)

if show_vis:
    visualize_codematrix_similarity(codemat)

img_filenames = sorted(os.listdir(img_dir))
imgs = []
for imgf in img_filenames:
    if not imgf.startswith('image'):
        continue
    p = os.path.join(img_dir, imgf)
# for p in img_paths:
    img = normalize_image(cv2.imread(p)[:,:,0])
    imgs.append(img)

# load white and black
# white = normalize_image(cv2.imread(white_path)[:,:,0])
# black = normalize_image(cv2.imread(black_path)[:,:,0])
# albedo = white - black
# ambient = black

# if remove_albamb:
#     for i in range(len(imgs)):
#         imgs[i] = (imgs[i] - ambient) / albedo
        # cv2.imwrite(os.path.join(datadir, "noalbamb%04d.png"%(i)), ((imgs[i].clip(0, 1)) * 255).astype(np.uint8))

gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:,:,0].astype(np.float32) * 1000  # turn into mm.

Kp = compute_blender_projector_intrinsic(
    800, 600, 0.838, 0.629, reverse_y=False
)
Kc = compute_ideal_intrinsic(
    50, 800, 600, 36, 27, reverse_y=False
)
R = np.eye(3)
T = np.array([200, 0, 0])
# Kp, Kc, R, T = hdsetting.proj_intri, hdsetting.cam_intri, hdsetting.R, hdsetting.T

gt_coresponding = compute_coresponding_from_depth(gt_depth, Kc, Kp, R, T)
# print(gt_depth[300:400, 400:500])
# print(gt_coresponding[300:400, 400:500])

SL = SLPipeline(Kp, Kc, R, T, codemat)
img_packed = SL.images_to_code_array(imgs)
if p_zncc==0:
    matched_indices = SL.match(img_packed)
else:
    matched_indices = SL.match(img_packed, lambda a,b: ZNCC_p_np(a, b, p_zncc, torch.device("cuda")))
print(matched_indices[300, 400])

# disparity map
h, w = np.mgrid[:600, :800]
disparity = matched_indices - w
# error_mask = disparity < 0
# matched_indices[error_mask] = 0
if show_vis:
    visualize_disparity(disparity)

# gt disparity
gt_disparity = gt_coresponding - w
if show_vis:
    visualize_disparity(gt_disparity)
    visualize_errormap(matched_indices, gt_coresponding, 'log10')

depth = SL.get_depth(matched_indices)
print(depth[300, 400])
print(gt_depth[300, 400])
if show_vis:
    visualize_depth(gt_depth, vmin=1200, vmax=2500)
    visualize_depth(depth, vmin=1200, vmax=2500)
print(gt_depth.min(), gt_depth.max())
print(depth.min(), depth.max())