import os
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

from SLPipeline.pipeline import SLPipeline
from SLPipeline.utils import *

dbg_dir = "Simulator/testdir/debug/"
code_matrix_path = "Simulator/testdir/code_matrix.png"
img_dir = "Simulator/testdir/output/"
depth_path = "Simulator/testdir/output/depth0000.exr"

codemat = normalize_image(cv2.imread(code_matrix_path)[:,:,0])

img_filenames = sorted(os.listdir(img_dir))
imgs = []
for imgf in img_filenames:
    if imgf.startswith('depth'):
        continue
    p = os.path.join(img_dir, imgf)
    img = normalize_image(cv2.imread(p)[:,:,0])
    imgs.append(img)

gt_depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:,:,0].astype(np.float32) * 1000  # turn into mm.

Kp = compute_blender_projector_intrinsic(
    800, 600, 0.838, 0.629
)
Kc = compute_ideal_intrinsic(
    50, 800, 600, 36, 27
)
R = np.eye(3)
T = np.array([200, 0, 0])

SL = SLPipeline(Kp, Kc, R, T, codemat)
img_packed = SL.images_to_code_array(imgs)
matched_indices = SL.match(img_packed)
print(matched_indices[300, 400])

# disparity map
h, w = np.mgrid[:600, :800]
disparity = matched_indices - w
error_mask = disparity < 0
matched_indices[error_mask] = 0
visualize_disparity(disparity)

depth = SL.get_depth(matched_indices)
print(depth[300, 400])
print(gt_depth[300, 400]* 1000)
visualize_depth(gt_depth)
visualize_depth(depth)

print(depth.min(), depth.max())