import os
import sys
import json
import numpy as np
import subprocess
from pathlib import Path

class BlenderSubprocess:
    def __init__(self, exe_path, scene_path, pattern_path, output_path, script_path, cwd) -> None:
        self.exe_path = exe_path
        self.scene_path = scene_path
        self.pattern_path = pattern_path
        self.output_path = output_path
        self.script_path = script_path
        self.cwd_path = cwd

        self.proc:subprocess.Popen = None

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