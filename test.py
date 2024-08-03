import os
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import OpenEXR

from SLPipeline.pipeline import SLPipeline
from SLPipeline.utils import *

code_matrix_path = "Simulator/testdir/code_matrix.png"
img_dir = "Simulator/testdir/output/"
depth_path = "Simulator/testdir/output/depth0000.exr"

codemat = normalize_image(cv2.imread(code_matrix_path))

img_filenames = sorted(os.listdir(img_dir))
imgs = []
for imgf in img_filenames:
    if imgf.startswith('depth'):
        continue
    p = os.path.join(img_dir, imgf)
    img = normalize_image(cv2.imread(p))
    imgs.append(img)

depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

Kc = compute_ideal_intrinsic(
    50, 800, 600, 36, 27
)
R = np.eye(3)
T = np.array([])
