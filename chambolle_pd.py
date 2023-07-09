"""
Realizes the primal-dual algorithm proposed in 
"A first-order primal-dual algorithm for convex
problems with applications to imaging", A.Chambolle,2010 
with both gray-scale and color image.
Implementation detail from 
"Chambolle's Projection Algorithm for Total Variation
Denoising", IPOL 2013.
adaptation from: 
https://github.com/crowsonkb/tv-denoise/blob/master/tv_denoise/chambolle.py

"""

import numpy as np
import cv2
from utils.utils import signaltonoise, grad, div, magnitude


def tv_denoise_chambolle(image, weight, step_size=0.25, N = 100, tol=1e-10):
    """
    Total variation image denoising with Chambolle's projection algorithm.
    Note that lambd := 1/lambda from the original algorithm. 
    The TV scheme was not specified in the paper
    "Chambolle's Projection Algorithm for Total Variation Denoising"
    diff = np.max(magnitude(new_p - p)) is the original stopping criterion
    prescribed by Chambolle. We forsake it for SNR due to algorithm comparison.
    
    image: noisy image, type np.array
    weight: weight parameter between value 10-20.
    step_size: projection step size. convergence is guaranteed for step size under 0.25.
    N: max iter.
    tol: stop iteration when SNR_{i}-SNR_{i-1}<tol.
    """
    lambd = 1/weight
    image = image.astype(np.float32)
    # print(image)
    image = np.atleast_3d(image)
    p = np.zeros((2,) + image.shape, image.dtype)
    # print(p.shape)
    image_over_strength = image*lambd
    snr = float(signaltonoise(image))
    snr_lst = [snr, ]
    for i in range(N):
        grad_div_p_i = grad(div(p) - image_over_strength)
        # print(grad_div_p_i.shape)
        mag_gdpi = magnitude(grad_div_p_i, axis=(0, -1), keepdims=True)
        # print(mag_gdpi.shape)
        # x = image - div(p)*(1/lambd)
        new_p = (p + step_size * grad_div_p_i) / (1 + step_size * mag_gdpi)
        # diff = np.max(magnitude(new_p - p))
        x_new = image - (1/lambd) * div(new_p)
        
        snr = float(signaltonoise(x_new))
        if snr-snr_lst[-1]<tol:
            break
        snr_lst.append(snr)
        p[:] = new_p
        print(snr)
        # if i%5 == 0:

        x_temp = x_new.astype(np.uint8)
        os_dir = [str(i),"temp.npy"]
        os_dir = "_".join(os_dir)
        np.save(os_dir, np.squeeze(x_temp))
        
        print(i)
    
    return np.squeeze(image - (1/lambd) * div(p)), i