import os
import numpy as np
import json
from enum import Enum

class HardwareSettings:
    class SystemType(Enum):
        normal = 0   # a normal projector and a normal camera that both can be modeled by pinhole model

    def __init__(self, hardware_config_file:str=None,
                 type: SystemType=SystemType.normal, 
                 cam_intri: np.ndarray = None, 
                 proj_intri: np.ndarray = None, 
                 R: np.ndarray = None, T: np.ndarray = None):
        '''
        cam_intri:  the intrinsic matrix of the camera, (3, 3) (turn the 3D point coordinates (in mm) in camera sapce to 2D pixel coordinates)
        proj_intri:  the intrinsic matrix of the projector, (3, 3)
        R:  the rotation matrix of the camera relative to the projector, (3, 3)
        T:  the translation vector of the camera relative to the projector, (3) (in mm)
        '''
        if hardware_config_file is not None:
            self.setup_from_config(hardware_config_file)
        else:
            self.type = type
            self.cam_intri = cam_intri
            self.proj_intri = proj_intri
            self.R = R
            self.T = T
        
    def setup_from_config(self, hardware_config_file):
        with open(hardware_config_file, 'r') as f:
            d = json.load(f)
        self.type = HardwareSettings.SystemType[d['type']]
        self.cam_intri = np.array(d['camera']['intrisic'])   # (3, 3)
        self.proj_intri = np.array(d['projector']['intrisic'])   # (3, 3)
        self.R = np.array(d['R'])   # (3, 3)
        self.T = np.array(d['T'])   # (3)

    def get_config_dict(self):
        d = {
            'type': self.type.name,
            'camera': {
                'intrisic': self.cam_intri.tolist(),
            },
            'projector': {
                'intrisic': self.proj_intri.tolist(),
            },
            'R': self.R.tolist(),
            'T': self.T.tolist()
        }
        return d

    
    def save_to_file(self, file_path:str):
        d = self.get_config_dict()
        with open(file_path, 'w') as f:
            json.dump(d, f, indent=4)