import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2
import time
from utils.utils import D, DT

def FGP_gray2d(y, lambd, n_iters, tv_type='anisotropic', \
               ground_truth = None, eps = 1e-3):
    # Main loop
    n1, n2 = y.shape
    grad_next = np.zeros((n1, n2, 2))
    grad_prev = np.zeros((n1, n2, 2))
    u = np.zeros((n1, n2, 2))
    t_prev = 1
    tic = time.perf_counter()
    if ground_truth == None:
        # if the original is not given
        
        for i in range(n_iters):
            grad_next = u + 1 / (8 * lambd) * D(y - lambd * DT(u))
            deno = np.zeros((n1, n2, 2))
            
            if tv_type == 'anisotropic':
                deno[:, :, 0] = np.maximum(1, np.abs(grad_next[:, :, 0]))
                deno[:, :, 1] = np.maximum(1, np.abs(grad_next[:, :, 1]))
            else:
                deno[:, :, 0] = np.maximum(1, np.sqrt(grad_next[:, :, 0]**2 + grad_next[:, :, 1]**2))
                deno[:, :, 1] = deno[:, :, 0]
            
            grad_next /= deno
            t_next = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
            u = grad_next + (t_prev - 1) / t_next * (grad_next - grad_prev)
            x_temp = y - lambd*DT(grad_next)
            np.linalg.norm(lambd*DT(grad_next)-lambd*DT(grad_prev))/np.linalg.norm(x_temp)>eps
            grad_prev = grad_next
            t_prev = t_next
            
            
            
            # save denoising process as .npy file 
            x_temp = x_temp.astype(np.uint8)
            os_dir = [str(i),"temp.npy"]
            os_dir = "_".join(os_dir)
            np.save(os_dir, x_temp)

    else:
        # TODO: save obtained image
        k = 1
        while np.linalg.norm(ground_truth, x)/np.linalg.norm(x)>eps:
            grad_next = u + 1 / (8 * lambd) * D(y - lambd * DT(u))
            deno = np.zeros((n1, n2, 2))
            
            if tv_type == 'anisotropic':
                deno[:, :, 0] = np.maximum(1, np.abs(grad_next[:, :, 0]))
                deno[:, :, 1] = np.maximum(1, np.abs(grad_next[:, :, 1]))
            else:
                deno[:, :, 0] = np.maximum(1, np.sqrt(grad_next[:, :, 0]**2 + grad_next[:, :, 1]**2))
                deno[:, :, 1] = deno[:, :, 0]
            
            grad_next /= deno
            t_next = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
            u = grad_next + (t_prev - 1) / t_next * (grad_next - grad_prev)
            grad_prev = grad_next
            t_prev = t_next 
            k+=1
            if k > n_iters: 
                break
    x = y - lambd * DT(grad_next)
    toc = time.perf_counter()
    runtime = toc - tic 

    return x


def FGP_color(y, lambda_val, n_iters, tv_type='anisotropic',\
               ground_truth = None, eps = 1e-3):
    # TODO breaking criteria
    n1, n2, _ = y.shape
    grad_next = np.zeros((n1, n2, 3, 2))
    grad_prev = np.zeros((n1, n2, 3, 2))
    u = np.zeros((n1, n2, 3, 2))

    t_prev = 1

    if tv_type != 'isotropic' and tv_type != 'anisotropic':
        raise ValueError("Unknown tv_type (should be either 'isotropic' or 'anisotropic')")

    if len(lambda_val) == 1:
        lambda_val = [lambda_val] * 3
    elif len(lambda_val) != 3:
        raise ValueError("The length of lambda should be 1 or 3")

    if min(lambda_val) <= 0:
        raise ValueError("The input lambda should take positive values")

    if tv_type == 'anisotropic':
        for i in range(n_iters):
            grad_next = u + 1/8 * D(y - DT(u))
            deno = np.zeros((n1, n2, 3, 2))
            deno[:, :, 0, 0] = 1 / lambda_val[0] * np.maximum(lambda_val[0], np.abs(grad_next[:, :, 0, 0]))
            deno[:, :, 0, 1] = 1 / lambda_val[0] * np.maximum(lambda_val[0], np.abs(grad_next[:, :, 0, 1]))
            deno[:, :, 1, 0] = 1 / lambda_val[1] * np.maximum(lambda_val[1], np.abs(grad_next[:, :, 1, 0]))
            deno[:, :, 1, 1] = 1 / lambda_val[1] * np.maximum(lambda_val[1], np.abs(grad_next[:, :, 1, 1]))
            deno[:, :, 2, 0] = 1 / lambda_val[2] * np.maximum(lambda_val[2], np.abs(grad_next[:, :, 2, 0]))
            deno[:, :, 2, 1] = 1 / lambda_val[2] * np.maximum(lambda_val[2], np.abs(grad_next[:, :, 2, 1]))

            grad_next = grad_next / deno
            t_next = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
            u = grad_next + (t_prev - 1) / t_next * (grad_next - grad_prev)
            grad_prev = grad_next
            t_prev = t_next
            x_temp = y - DT(grad_next)
            # save denoising process as .npy file 
            x_temp = x_temp.astype(np.uint8)
            os_dir = [str(i),"temp.npy"]
            os_dir = "_".join(os_dir)
            np.save(os_dir, x_temp)

    else:
        for i in range(n_iters):
            grad_next = u + 1/8 * D(y - DT(u))
            deno = np.zeros((n1, n2, 3, 2))
            deno[:, :, 0, 0] = 1/lambda_val[0] * np.maximum(lambda_val[0], np.sqrt(grad_next[:, :, 0, 0]**2 + grad_next[:, :, 0, 1]**2))
            deno[:, :, 0, 1] = deno[:, :, 0, 0]
            deno[:, :, 1, 0] = 1/lambda_val[1] * np.maximum(lambda_val[1], np.sqrt(grad_next[:, :, 1, 0]**2 + grad_next[:, :, 1, 1]**2))
            deno[:, :, 1, 1] = deno[:, :, 1, 0]
            deno[:, :, 2, 0] = 1/lambda_val[2] * np.maximum(lambda_val[2], np.sqrt(grad_next[:, :, 2, 0]**2 + grad_next[:, :, 2, 1]**2))
            deno[:, :, 2, 1] = deno[:, :, 2, 0]

            grad_next = grad_next / deno
            t_next = (1 + np.sqrt(1 + 4 * t_prev**2)) / 2
            u = grad_next + (t_prev - 1) / t_next * (grad_next - grad_prev)
            grad_prev = grad_next
            t_prev = t_next   
            x_temp = y - DT(grad_next)
            # save denoising process as .npy file 
            x_temp = x_temp.astype(np.uint8)
            os_dir = [str(i),"temp.npy"]
            os_dir = "_".join(os_dir)
            np.save(os_dir, x_temp)


    x = y - DT(grad_next)  # convert to the primal optimal
    return x

def FGP(u0, lambda_val, n_iters, isotropic = True \
        ,channel_axis = None, eps = 1e-3):
    if channel_axis == True: # color image
        assert len(u0.shape)==3, "dimension mismatch"
        # color image
        if isotropic == True:
            tv_type = "isotropic"
        else:
            tv_type = "anisotropic"
        out = FGP_color(u0,lambda_val, n_iters, \
            tv_type = tv_type,
                eps = eps)#, channel_axis = True)

    else:
        assert len(u0.shape)==2, "dimension mismatch"
        if isotropic == True:
            tv_type = "isotropic"
        else:
            tv_type = "anisotropic"
        out = FGP_gray2d(u0,lambda_val, n_iters, \
              tv_type = tv_type,
                eps = eps)
    return out
    
