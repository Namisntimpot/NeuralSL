import numpy as np
import cv2

def ZNCC(a:np.ndarray, b:np.ndarray):
    '''
    a: (..., h, w_img, k), image code array.
    b: (..., w_pat, k), pattern code matrix.
    the result (h, p, q) means ZNCC(pixel(h, p), code(q))
    '''
    za = a - np.mean(a, axis=-1, keepdims=True)
    zb = b - np.mean(b, axis=-2, keepdims=True)
    l2_za = np.sqrt(np.sum(za ** 2, axis=-1, keepdims=True))
    l2_zb = np.sqrt(np.sum(zb ** 2, axis=-2, keepdims=True))
    return np.einsum("...pk, ...kq -> ...pq", za / l2_za, zb / l2_zb)
#     '''
#     the code vector should be at the last dim.
#     assuming a is the image code arrays (h, p, k), b is the pattern's code matrix (q, k), then the result (h, p, q) means ZNCC(pixel(h, p), code(q))
#     '''
#     za = a - np.mean(a, axis=-1, keepdims=True)
#     zb = b - np.mean(b, axis=-1, keepdims=True)
#     l2_za = np.sqrt(np.sum(za ** 2, axis=-1, keepdims=True))
#     l2_zb = np.sqrt(np.sum(zb ** 2, axis=-1, keepdims=True))
#     ret = np.einsum("...pk, ...qk -> ...pq", za / l2_za, zb / l2_zb)
#     return ret


def compute_ideal_intrinsic(focal_length, reso_x, reso_y, sensor_x, sensor_y):
    '''
    Compute an ideal intrinsic matrix  (camera coordinate -> pixel coordinate)
    use mm as the unit
    '''
    pixel_rho_x = reso_x / sensor_x
    pixel_rho_y = reso_y / sensor_y
    cx, cy = reso_x / 2, reso_y / 2
    return np.array(
        [
            [pixel_rho_x * focal_length, 0, pixel_rho_x * cx],
            [0, -pixel_rho_y * focal_length, pixel_rho_y * cy],
            [0, 0, 1]
        ]
    )


def normalize_image(img:np.ndarray, bit_depth: int = 8):
    max_val = 2 ** bit_depth - 1
    return img.astype(np.float32) / max_val