import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2
import time
from utils.utils import D, Div, tv_norm,signaltonoise 

"""realizes algorithm proposed by:
"""

def FGP_gray2d(y, lambd, n_iters, tv_type='anisotropic', \
               ground_truth = None, eps = 1e-5):
    # Main loop
    n1, n2 = y.shape
    grad_next = np.zeros((n1, n2, 2))
    grad_prev = np.zeros((n1, n2, 2))
    u = np.zeros((n1, n2, 2))
    t_prev = 1
    tic = time.perf_counter()
    
    snr_lst = [float(signaltonoise(y)),]
    for i in range(n_iters):
        grad_next = u + 1 / (8 * lambd) * D(y - lambd * Div(u))
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
        x_temp = y - lambd*Div(grad_next)
        snr = float(signaltonoise(x_temp))
        if snr-snr_lst[-1]<eps:
            break
        snr_lst.append(snr)
        # if np.linalg.norm(lambd*Div(grad_next)-lambd*Div(grad_prev))/np.linalg.norm(x_temp)<eps:
        #     break
        grad_prev = grad_next
        t_prev = t_next
        
        
        # save denoising process as .npy file 
        x_temp = x_temp.astype(np.uint8)
        os_dir = [str(i),"temp.npy"]
        os_dir = "_".join(os_dir)
        np.save(os_dir, x_temp)

    x = y - lambd * Div(grad_next)
    toc = time.perf_counter()
    runtime = toc - tic 

    return x, i


def FGP_color(y, lambda_val, n_iters, tv_type='anisotropic',\
               ground_truth = None, eps = 1e-5):
    # TODO breaking criteria
    n1, n2, _ = y.shape
    grad_next = np.zeros((n1, n2, 3, 2))
    grad_prev = np.zeros((n1, n2, 3, 2))
    u = np.zeros((n1, n2, 3, 2))

    t_prev = 1
    val_lst = [float(signaltonoise(y)),]

    if tv_type != 'isotropic' and tv_type != 'anisotropic':
        raise ValueError("Unknown tv_type (should be either 'isotropic' or 'anisotropic')")

    if type(lambda_val) != "list":
        lambda_val = [lambda_val] * 3
    elif len(lambda_val) != 3:
        raise ValueError("The length of lambda should be 1 or 3")

    if min(lambda_val) <= 0:
        raise ValueError("The input lambda should take positive values")
    if tv_type == 'anisotropic':
        for i in range(n_iters):
            grad_next = u + 1/8 * D(y - Div(u))
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
            x_temp = y - Div(grad_next)
            # val_temp  = np.linalg.norm(Div(grad_prev)-Div(grad_next))/np.linalg.norm(x_temp)
            # val_temp = tv_norm(x_temp)
            val_temp = float(signaltonoise(x_temp))
            # print(val_temp)
            if val_temp-val_lst[-1]<eps:
                break
            print(val_temp)

            # if val_temp<eps:
            #     break
            val_lst.append(val_temp)
            grad_prev = grad_next
            t_prev = t_next
            
            # save denoising process as .npy file 
            x_temp = x_temp.astype(np.uint8)
            os_dir = [str(i),"temp.npy"]
            os_dir = "_".join(os_dir)
            np.save(os_dir, x_temp)

    else:
        for i in range(n_iters):
            grad_next = u + 1/8 * D(y - Div(u))
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
            x_temp = y - Div(grad_next)
            # val_temp  = np.linalg.norm(Div(grad_prev)-Div(grad_next))/np.linalg.norm(x_temp)
            # val_temp = tv_norm(x_temp)
            # print(val_temp)
            val_temp = float(signaltonoise(x_temp))
            # print(val_temp)
            if val_temp-val_lst[-1]<eps:
                break
            print(val_temp)
            # if val_temp<eps:
            #     break
            val_lst.append(val_temp)
            grad_prev = grad_next
            t_prev = t_next   
            
            # save denoising process as .npy file 
            x_temp = x_temp.astype(np.uint8)
            os_dir = [str(i),"temp.npy"]
            os_dir = "_".join(os_dir)
            np.save(os_dir, x_temp)


    x = y - Div(grad_next)  # convert to the primal optimal
    return x, i #val_lst

def FGP(u0, lambda_val, n_iters, isotropic = True \
        ,channel_axis = None, eps = 1e-6):
    i = 0
    if channel_axis == True: # color image
        assert len(u0.shape)==3, "dimension mismatch"
        # color image
        if isotropic == True:
            tv_type = "isotropic"
        else:
            tv_type = "anisotropic"
        out, i = FGP_color(u0,lambda_val, n_iters, \
            tv_type = tv_type,
                eps = eps)#, channel_axis = True)

    else:
        assert len(u0.shape)==2, "dimension mismatch"
        if isotropic == True:
            tv_type = "isotropic"
        else:
            tv_type = "anisotropic"
        out, i = FGP_gray2d(u0,lambda_val, n_iters, \
              tv_type = tv_type,
                eps = eps)
    return out, i
    
