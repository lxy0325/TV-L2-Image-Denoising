""" ADMM color test. """
import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2
import time
from utils.utils import D
from utils.utils import Div
from utils.utils import tv_norm, signaltonoise

"""
We give different convergence criteria for color images due to the general failure of implementing
the usual normalized_norm<eps one. (image channel size would change eps value drastically.) 
We compute the TV/SNR for each iteration and break from iteration 
once the TV/SNR starts to increase/decrease. According to:
http://www.corc.ieor.columbia.edu/reports/techreports/tr-2004-03.pdf
"""


# Auxiliary functions
def z_solver(x, u, lambd, rho, mask, tv_type):
    w = mask * (D(x) + (1 / rho) * u)
    if tv_type == 'anisotropic':
        z = soft_threshold(w, lambd / rho)
    else:
        w_v, w_h = w[:, :, 0], w[:, :, 1]
        t = np.sqrt(w_v**2 + w_h**2)
        z = (np.atleast_3d(soft_threshold(t, lambd / rho) / (t + np.finfo(float).eps)))*w
    return z

def soft_threshold(x, kappa):
    return np.maximum(x - kappa, 0) - np.maximum(-x - kappa, 0)

def x_solver(z, u, u0, rho, deno):
    x = ifft2(fft2(u0 + rho * Div(z) - Div(u)) / deno)
    return x

"""implements ADMM for TV-L^2."""
def ADMM_3D(u0,lambd, N, \
              tv_type = "anisotropic",
                rho = 0.05, mu = 10, tau = 2, ground_truth = None, eps = 1e-3, channel_axis = None):
    # note: when ground truth is nonzero, eps is used to
    # ensure the quality of output image
    m, n, c = u0.shape
    assert c == 3, "color channel mismatch error."
    # Initialization
    u0_R = u0[:,:,0]
    u0_G = u0[:,:,1]
    u0_B = u0[:,:,2]
    x = np.zeros_like(u0)
    x_next = np.zeros_like(u0)
    z = np.zeros((m,n,3,2))
    z_next = np.zeros((m,n,3,2))
    u = np.zeros_like(z)

    # z = D(u_channel)
    mask = np.ones((m,n,2))
    mask[-1, :, 0] = 0
    mask[:, -1, 1] = 0
    dh = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    dh_pad = np.zeros((m, n))
    dh_pad[m//2:m//2+3, n//2:n//2+3] = dh 
    # horizontal difference operator
    dv = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    dv_pad = np.zeros((m, n))
    dv_pad[m//2:m//2+3, n//2:n//2+3] = dv
    # vertical difference operator
    fdh = fft2(dh_pad)
    fdv = fft2(dv_pad)
    deno = 1 + rho * np.abs(fdh)**2 + rho * np.abs(fdv)**2
        
    z = D(u0)
    print(z.shape)
    # print("z=",z)
    print("dtz=",Div(z))
    # tic = time.perf_counter()
    # denoise_process = []
    # Main loop
    # np.save("0_temp.npy",u0)
    """To display the denoising process, the intermediate values
    are saved to a temp.npy file. """
    # val_lst = [tv_norm(u0),]
    val_lst = [signaltonoise(u0),]
    # store the TV norm of images for iteration control
    # iterate for a set number of times
    for i in range(N):
        # TODO: set stopping criteria with original image
        x_next[:,:,0] = x_solver(z[:,:,0,:], u[:,:,0,:], u0_R, rho, deno)
        z_next[:,:,0,:] = z_solver(x_next[:,:,0], u[:,:,0,:], lambd, rho, mask, tv_type)
        x_next[:,:,1] = x_solver(z[:,:,1,:], u[:,:,1,:], u0_G, rho, deno)
        z_next[:,:,1,:] = z_solver(x_next[:,:,1], u[:,:,1,:], lambd, rho, mask, tv_type)
        x_next[:,:,2] = x_solver(z[:,:,2,:], u[:,:,2,:], u0_B, rho, deno)
        z_next[:,:,2,:] = z_solver(x_next[:,:,2], u[:,:,2,:], lambd, rho, mask, tv_type)

        u_next = u + rho * (D(x_next) - z_next)
        # print(DT(z))
        # z_next = np.stack([z_next_R, z_next_G, z_next_B], axis = 2)
        # print(DT(z_next))
        s = -rho * (Div(z_next - z))
        r = D(x) - z
        s_norm = np.linalg.norm(s)
        r_norm = np.linalg.norm(r)
        if r_norm > mu * s_norm:
            rho = rho * tau
        elif s_norm > mu * r_norm:
            rho = rho / tau
        print("ite",i)
        # print(np.linalg.norm(x_next-x, axis = 1))
        # TODO: find out why does the usual convergence not work
        
        # val_temp = tv_norm(x_next)
        val_temp = float(signaltonoise(x_next))
        if val_temp-val_lst[-1]<eps:
            # TV is increasing
            break
        val_lst.append(val_temp)

        x = x_next
        z = z_next
        u = u_next

        x_temp = x.astype(np.uint8)
        os_dir = [str(i),"temp.npy"]
        os_dir = "_".join(os_dir)
        np.save(os_dir, x_temp)
        # denoise_process.append(u)

    # toc = time.perf_counter()
    # runtime = toc - tic
    
    return x, i