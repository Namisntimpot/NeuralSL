import os
import sys
import json
import numpy as np
import subprocess

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
