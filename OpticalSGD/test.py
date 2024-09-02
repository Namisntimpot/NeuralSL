import numpy as np
from SLPipeline.utils import *
from SLPipeline.hardware_settings import HardwareSettings

Kp = compute_blender_projector_intrinsic(
    800, 600, 0.838, 0.629
)
Kc = compute_ideal_intrinsic(
    50, 800, 600, 36, 27
)
R = np.eye(3)
T = np.array([50, 0, 0])

hardware = HardwareSettings(None, HardwareSettings.SystemType.normal, Kc, Kp, R, T)
hardware.save_to_file("OpticalSGD/testimg/hardware_settings.json")