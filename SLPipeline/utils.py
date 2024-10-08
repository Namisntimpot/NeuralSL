import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def ZNCC(a:np.ndarray, b:np.ndarray):
    '''
    a: (..., h, w_img, k), image code array.
    b: (k, w_pat), pattern code matrix.
    the result (h, p, q) means ZNCC(pixel(h, p), code(q))
    '''
    za = a - np.mean(a, axis=-1, keepdims=True) + 1e-6
    zb = b - np.mean(b, axis=-2, keepdims=True) + 1e-6
    l2_za = np.sqrt(np.sum(za ** 2, axis=-1, keepdims=True))
    l2_zb = np.sqrt(np.sum(zb ** 2, axis=-2, keepdims=True))
    return np.einsum("...pk, ...kq -> ...pq", za / l2_za, zb / l2_zb)


def ZNCC_torch(a:torch.Tensor, b:torch.Tensor):
    za = a - torch.mean(a, dim=-1, keepdim=True) + 1e-6
    zb = b - torch.mean(b, dim=-2, keepdim=True) + 1e-6
    l2_za = torch.sqrt(torch.sum(za ** 2, dim=-1, keepdim=True))
    l2_zb = torch.sqrt(torch.sum(zb ** 2, dim=-2, keepdim=True))
    return torch.einsum("...pk, ...kq -> ...pq", za / l2_za, zb / l2_zb)

def compute_ideal_intrinsic(focal_length, reso_x, reso_y, sensor_x, sensor_y, reverse_y:bool = False):
    '''
    Compute an ideal intrinsic matrix  (camera coordinate -> pixel coordinate)
    use mm as the unit
    '''
    pixel_rho_x = reso_x / sensor_x
    pixel_rho_y = reso_y / sensor_y
    cx, cy = sensor_x / 2, sensor_y / 2
    return np.array(
        [
            [pixel_rho_x * focal_length, 0, pixel_rho_x * cx],
            [0, pixel_rho_y * focal_length * (-1 if reverse_y else 1), pixel_rho_y * cy],
            [0, 0, 1]
        ]
    )

def compute_blender_projector_intrinsic(reso_x, reso_y, scale_x, scale_y, reverse_y:bool = False):
    '''
    reso: the resolutionos the pattern used.
    scale: the scale value of of the second Mapping node of the spot light in blender

    当reverse_y == False, 所用的相机坐标系是一般的，x左y下、相机拍摄z正方向, 右手系.
    '''
    return np.array(
        [
            [reso_x / scale_x, 0, 0.5 * reso_x],
            [0, reso_y / scale_y * (-1 if reverse_y else 1), 0.5 * reso_y],  # ??是否要加负号？
            [0, 0, 1]
        ]
    )

def compute_rotation_matrix(rot_x, rot_y, rot_z, order = 'xyz', eps=1e-8):
    '''
    compute rotation matrix from euler engles. rot_x, rot_y, rot_z are rotation angles in degree.  
    for blender's camera, the order should be 'xyz'
    '''
    from math import pi
    from scipy.spatial.transform import Rotation as R

    # 定义欧拉角（单位是弧度）
    angles = [rot_x * pi / 180, rot_y * pi / 180, rot_z * pi / 180]  # [X轴旋转, Y轴旋转, Z轴旋转]

    # 指定旋转顺序，例如 'xyz'
    r = R.from_euler('xyz', angles)

    # 将欧拉角转换为旋转矩阵
    rotation_matrix = r.as_matrix()
    rotation_matrix[rotation_matrix < eps] = 0
    return rotation_matrix
    

def decomposite_instrisic(intri:np.ndarray):
    '''
    return fx, cx, fy, cy in pixel unit
    '''
    return intri[0, 0], intri[0, 2], abs(intri[1, 1]), intri[1, 2]


def compute_coresponding_from_depth(depth_map:np.ndarray, cam_intri:np.ndarray, proj_intri:np.ndarray, R:np.ndarray, T:np.ndarray):
    '''
    warning: the unit issue! (now hdsettings use mm as length unit)
    with the real depth map (in mm), camera and projector's intrisic matrix known, compute the real coresponding map.  
    R:  the rotation matrix of the camera relative to the projector, (3, 3)
    T:  the translation vector of the camera relative to the projector, (3) (in mm)
    it's a re-projection process.  
    the depth here is defined as the projected distance along the z-axis, rather than the distance from the camera's optical center to the object point.
    '''
    h, w = depth_map.shape
    y, x = np.mgrid[:h, :w]
    # camera's pixel coordinates
    pixel_coord_cam = np.stack([x, y, np.ones_like(x)], axis=-1)  # (h, w, 3)
    cam_intri_inv = np.linalg.inv(cam_intri)  # (3, 3)
    space_coord_cam = np.einsum("mk, hwk -> hwm", cam_intri_inv, pixel_coord_cam * depth_map[:,:,np.newaxis])
    space_coord_cam_ho = np.concatenate([space_coord_cam, np.ones_like(space_coord_cam[:,:,:1])] , axis=-1)  # (h, w, 4)
    trans = RT2TransformMatrix(R, T, want='3x4')
    space_coord_proj = np.einsum("jk, hwk -> hwj", trans, space_coord_cam_ho)  # (h, w, 3)
    pixel_coord_proj = np.einsum("jk, hwk -> hwj", proj_intri, space_coord_proj) / (space_coord_proj[...,2:3] + 1e-6)  # (h, w, 3)
    # y坐标是匹配到的结果.
    return pixel_coord_proj[...,0]

def RT2TransformMatrix(R:np.ndarray, T:np.ndarray, want:str = '3x4'):
    '''
    return [R, T // 0, 1]
    '''
    m3x4 = np.concatenate([R, T[:, np.newaxis]], axis=-1)
    if want == '3x4':
        return m3x4
    else:
        m4x4 = np.concatenate([m3x4, np.array([[0, 0, 0, 1]])], axis=0)
        return m4x4


def normalize_image(img:np.ndarray, bit_depth: int = 8):
    max_val = 2 ** bit_depth - 1
    return img.astype(np.float32) / max_val


def visualize_disparity(disparity: np.ndarray, vmin = 0, vmax = 200, save_path:str = None):
    h, w = disparity.shape[-2], disparity.shape[-1]
    mask = disparity < 0   # 异常
    disparity[mask] = 0
    
    plt.figure(figsize=(10, 6))
    plt.imshow(disparity, cmap='plasma', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Color Map')
    plt.title('Disparity Map')
    plt.xlabel('W')
    plt.ylabel('H')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def visualize_depth(depth: np.ndarray, vmin = 0, vmax = 4000, unit:str = 'mm', save_path:str = None):
    plt.figure(figsize=(10, 6))
    plt.imshow(depth, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label=f'Color Map ({unit})')
    plt.title('Depth Map')
    plt.xlabel('W')
    plt.ylabel('H')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def visualize_codematrix_similarity(codes:np.ndarray, save_path = None):
    zncc = ZNCC(codes.T, codes)
    plt.figure(figsize=(10, 6))
    plt.imshow(zncc, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(label='Color Map')
    plt.title('Codes Similarity')

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def visualize_codematrix(codes:np.ndarray, save_path = None):
    k, w = codes.shape
    h = k * 100
    vis = np.zeros((h, w), dtype=codes.dtype)
    for i in range(k):
        vis[i*100 : (i+1)*100, :] = vis[i, :]
    if save_path is not None:
        cv2.imwrite(save_path, vis)
    return vis


def visualize_errormap(a, gt, bounds = 'log10', save_path = None, linthresh = 1, linscale = 1):
    if bounds == 'log10':
        bounds = [-100, -10, -1, 1, 10, 100]
    elif type(bounds) == list:
        pass
    else:
        raise NotImplementedError
    err = a - gt
    import matplotlib.colors as mcolors
    norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=-100, vmax=100, base=10)
    # 使用自定义颜色映射
    cmap = plt.get_cmap('bwr')  # 颜色映射选择
    fig, ax = plt.subplots()

    # 显示数据
    cax = ax.imshow(err, cmap=cmap, norm=norm)

    # 添加颜色条，并指定标签和比例
    cbar = fig.colorbar(cax)
    cbar.set_label('error (pixels)')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
    plt.close()


def load_cv2_for_exr():
    import sys, os
    import importlib
    if "cv2" in sys.modules:
        if not 'OPENCV_IO_ENABLE_OPENEXR' in os.environ:
            os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
            importlib.reload(sys.modules['cv2'])
        else:
            return
    else:
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
        importlib.import_module("cv2")