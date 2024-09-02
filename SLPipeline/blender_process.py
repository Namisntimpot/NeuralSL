import os
import sys
import json
import numpy as np
import subprocess
from pathlib import Path
import glob
import cv2

from SLPipeline.utils import normalize_image


class BlenderSubprocess:
    def __init__(self, exe_path, scene_path, pattern_path, output_path, script_path, cwd) -> None:
        self.exe_path = exe_path
        self.scene_path = scene_path
        self.pattern_path = pattern_path
        self.output_path = output_path
        self.script_path = script_path
        self.cwd_path = cwd

        self.proc:subprocess.Popen = None
        self.load_exr_tried = False

    def run_and_wait(self):
        self.proc = subprocess.Popen(args=[self.exe_path, "--background", self.scene_path,
                                            "--python", self.script_path, "--", "--render",
                                            "--pattern-dir", self.pattern_path, "--output-dir", self.output_path],
                                     cwd=self.cwd_path)
        return self.proc.wait()

    def resolve_path(self, path:str):
        p = Path(path)
        if p.is_absolute():
            return path
        cwd = Path(self.cwd_path)
        return cwd.joinpath(p).resolve().as_posix()
    
    def get_pattern_path(self):
        return self.resolve_path(self.pattern_path)

    def get_output_path(self):
        return self.resolve_path(self.output_path)

    def get_scene_path(self):
        return self.resolve_path(self.scene_path)
    
    def load_rendered_depth(self):
        '''
        return: depth: (h, w), numpy ndarray. dtype = np.float32
        '''
        outpath = self.get_output_path()
        fp = glob.glob(os.path.join(outpath, "depth*"))[0]
        # 如果是第一次调用，判断cv2的引入和os.environ['OPENCV_IO_ENABLE_OPENEXR']的情况
        if not self.load_exr_tried:
            self.load_exr_tried = True
            from SLPipeline.utils import load_cv2_for_exr
            load_cv2_for_exr()
        depth = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        return depth
    
    def load_rendered_images(self):
        '''
        return img (h, w, k), k is the number of rendering results (i.e. the number of patterns). numpy.ndarray, dtype = np.float32, range [0, 1]
        '''
        outpath = self.get_output_path()
        paths = sorted(glob.glob(os.path.join(outpath, "image*")))
        imgs = []
        for p in paths:
            p = Path(p)
            if p.suffix.lower() in ['.jpg', '.png']:
                im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if len(im.shape) != 2:
                    raise NotImplementedError("Only support grayscale image for now")
                im = normalize_image(im, bit_depth=8)
                imgs.append(im)
            else:
                raise NotImplementedError
        return np.stack(imgs, axis=-1)