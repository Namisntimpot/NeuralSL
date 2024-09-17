import os
import numpy as np
import json
from enum import Enum

from SLPipeline.utils import RT2TransformMatrix

class HardwareSettings:
    unit_dict = {
        'm':1000, 'mm':1
    }
    class SystemType(Enum):
        normal = 0   # a normal projector and a normal camera that both can be modeled by pinhole model

    def __init__(self, hardware_config_file:str=None,
                 type: SystemType=SystemType.normal, 
                 cam_intri: np.ndarray = None, 
                 proj_intri: np.ndarray = None, 
                 R: np.ndarray = None, T: np.ndarray = None, 
                 c2w_R: np.ndarray = None, c2w_T: np.ndarray = None,
                 unit: str = 'm'):
        '''
        cam_intri:  the intrinsic matrix of the camera, (3, 3) (turn the 3D point coordinates (in mm) in camera sapce to 2D pixel coordinates)
        proj_intri:  the intrinsic matrix of the projector, (3, 3)
        R:  the rotation matrix of the camera relative to the projector, (3, 3)  (i.e. cam2proj transform matrix), it's necessary
        T:  the translation vector of the camera relative to the projector, (3) (in mm)  it's necessary
        c2w_R: the rotation matrix of the camera relative to the world coordinate, if None, regard camera coordinate as world coordinate
        c2w_T: the translation vector of the camera relative to the world coordinate, if None, regard camera coordinate as world coordinate
        unit: the unit of translation.

        注意，计算内参时所用的相机坐标系是一般的坐标系，即x左y下，相机拍摄z正方向；但c2w这些东西所指定的相机坐标系不一定是它，可能是opengl坐标系，x左y上，相机拍摄z负方向.  
        也就是内参和c2w所用的相机坐标系可能不同，但内参所用的相机坐标系一定是一般坐标系，x左y下、相机拍摄z正方向! 
        '''
        if hardware_config_file is not None:
            self.setup_from_config(hardware_config_file)
        else:
            assert unit in HardwareSettings.unit_dict
            self.unit = unit
            self.type = type
            self.cam_intri = cam_intri
            self.proj_intri = proj_intri
            assert R is not None and T is not None
            self.R = R  # 这个R实际上就是c2p_R
            self.T = T  # 这个T实际上就是c2p_T
            self.c2p = RT2TransformMatrix(R, T, want='4x4')
            self.c2w = RT2TransformMatrix(c2w_R if c2w_R is not None else np.eye(3), c2w_T if c2w_T is not None else np.zeros((3,)), want='4x4')
        self.p2c = np.linalg.inv(self.c2p)            
        self.p2w = self.c2w @ self.p2c
        self.w2c = np.linalg.inv(self.c2w)
        self.w2p = np.linalg.inv(self.p2w)            
        
    def setup_from_config(self, hardware_config_file):
        with open(hardware_config_file, 'r') as f:
            d = json.load(f)
        self.unit = d['unit']
        self.type = HardwareSettings.SystemType[d['type']]
        self.cam_intri = np.array(d['camera']['intrisic'])   # (3, 3)
        self.proj_intri = np.array(d['projector']['intrisic'])   # (3, 3)
        self.c2p = np.array(d['c2p'])
        self.R, self.T = self.c2p[:3,:3], self.c2p[:3, -1]
        self.c2w = np.array(d['c2w'])
        

    def get_config_dict(self):
        d = {
            'type': self.type.name,
            'unit': self.unit,
            'camera': {
                'intrisic': self.cam_intri.tolist(),
            },
            'projector': {
                'intrisic': self.proj_intri.tolist(),
            },
            'R': self.R.tolist(),
            'T': self.T.tolist(),
            'c2p':self.c2p.tolist(),
            'c2w':self.c2w.tolist(),
        }
        return d
    
    def get_R(self, property:str):
        return getattr(self, property)[:3, :3]
    
    def get_T(self, property:str):
        return getattr(self, property)[:3, -1]
    
    def save_to_file(self, file_path:str):
        d = self.get_config_dict()
        with open(file_path, 'w') as f:
            json.dump(d, f, indent=4)

    def convert_unit(self, u:str):
        assert u in HardwareSettings.unit_dict
        scale = HardwareSettings.unit_dict[self.unit] / HardwareSettings.unit_dict[u]
        self.unit = u
        self.T /= scale
        for i in [self.c2p, self.c2w, self.p2c, self.p2w, self.w2c, self.w2p]:
            i[:3, -1] * scale