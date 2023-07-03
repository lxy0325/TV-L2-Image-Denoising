"""
Realizes the primal-dual algorithm proposed in 
"A first-order primal-dual algorithm for convex
problems with applications to imaging", A.Chambolle,2010 
with both gray-scale and color image.
Implementation detail from 
"Chambolle's Projection Algorithm for Total Variation
Denoising", IPOL 2013.
from: 
https://github.com/crowsonkb/tv-denoise/blob/master/tv_denoise/chambolle.py

"""

import numpy as np
import cv2


class ChambolleDenoiseStatus:
    """A status object supplied to the callback specified in tv_denoise_chambolle()."""
    i: int
    diff: float

def grad(arr):
    """Computes the discrete gradient of an image with central differences."""
    out = np.zeros((2,) + arr.shape, arr.dtype)
    out[0, :-1, :, ...] = arr[1:, :, ...] - arr[:-1, :, ...]
    out[1, :, :-1, ...] = arr[:, 1:, ...] - arr[:, :-1, ...]
    return out


def div(arr):
    """Computes the discrete divergence of a vector array."""
    out = np.zeros_like(arr)
    out[0, 0, :, ...] = arr[0, 0, :, ...]
    out[0, -1, :, ...] = -arr[0, -2, :, ...]
    out[0, 1:-1, :, ...] = arr[0, 1:-1, :, ...] - arr[0, :-2, :, ...]
    out[1, :, 0, ...] = arr[1, :, 0, ...]
    out[1, :, -1, ...] = -arr[1, :, -2, ...]
    out[1, :, 1:-1, ...] = arr[1, :, 1:-1, ...] - arr[1, :, :-2, ...]
    return np.sum(out, axis=0)


def magnitude(arr, axis=0, keepdims=False):
    """Computes the element-wise magnitude of a vector array."""
    return np.sqrt(np.sum(arr**2, axis=axis, keepdims=keepdims))


def tv_denoise_chambolle(image, lambd, step_size=0.25, tol=1e-5, callback=None):
    """
    Total variation image denoising with Chambolle's projection algorithm.
    Note that lambd := 1/lambda from the original algorithm. 
    """
    image = image.astype(np.float32)
    print(image)
    image = np.atleast_3d(image)
    p = np.zeros((2,) + image.shape, image.dtype)
    # print(p.shape)
    image_over_strength = image / lambd
    diff = np.inf
    i = 0
    while diff > tol:
        i += 1
        grad_div_p_i = grad(div(p) - image_over_strength)
        # print(grad_div_p_i.shape)
        mag_gdpi = magnitude(grad_div_p_i, axis=(0, -1), keepdims=True)
        # print(mag_gdpi.shape)
        x = image - lambd * div(p)
        new_p = (p + step_size * grad_div_p_i) / (1 + step_size * mag_gdpi)
        # diff = np.max(magnitude(new_p - p))
        x_new = image - lambd * div(new_p)
        diff = np.linalg.norm(x-x_new)/\
            np.linalg.norm(x)
        p[:] = new_p
        print(diff)
        # if i%5 == 0:
        x_temp = x_new.astype(np.uint8)
        os_dir = [str(i),"temp.npy"]
        os_dir = "_".join(os_dir)
        np.save(os_dir, np.squeeze(x_temp))
        if callback is not None:
            callback(ChambolleDenoiseStatus(i, float(diff)))
        
        print(i)
    
    return np.squeeze(image - lambd * div(p))