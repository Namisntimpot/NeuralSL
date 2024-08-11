import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

def ZNCC(a:np.ndarray, b:np.ndarray):
    '''
    a: (..., h, w_img, k), image code array.
    b: (..., w_pat, k), pattern code matrix.
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

def compute_ideal_intrinsic(focal_length, reso_x, reso_y, sensor_x, sensor_y):
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
            [0, -pixel_rho_y * focal_length, pixel_rho_y * cy],
            [0, 0, 1]
        ]
    )

def compute_blender_projector_intrinsic(reso_x, reso_y, scale_x, scale_y):
    '''
    reso: the resolutionos the pattern used.
    scale: the scale value of of the second Mapping node of the spot light in blender
    '''
    return np.array(
        [
            [reso_x / scale_x, 0, 0.5 * reso_x],
            [0, reso_y / scale_y, 0.5 * reso_y],
            [0, 0, 1]
        ]
    )


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


def visualize_depth(depth: np.ndarray, vmin = 0, vmax = 4000, save_path:str = None):
    plt.figure(figsize=(10, 6))
    plt.imshow(depth, cmap='viridis', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Color Map (mm)')
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