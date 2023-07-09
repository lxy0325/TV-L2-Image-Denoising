import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2
import time
from utils.utils import D, Div, signaltonoise

"""
Note that image channel size would require drastically different eps value. 
We compute the SNR for each iteration and break from iteration 
once the SNR starts to increase. Algorithm according to:
https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
http://www.corc.ieor.columbia.edu/reports/techreports/tr-2004-03.pdf
adaptation from:
https://github.com/Yunhui-Gao/total-variation-image-denoising
"""


# Auxiliary functions for projection
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
def ADMM_3D(image,weight, N, \
              tv_type = "anisotropic",
                rho = 0.05, mu = 10, tau = 2, eps = 1e-8):
    """
    image: noisy image, type np.array
    weight: weight parameter between value 10-20.
    tv_type: ="isotropic" or "anisotropic".
    N: max iter.
    eps: stop iteration when SNR_{i}-SNR_{i-1}<tol.
    """
    # note: when ground truth is nonzero, eps is used to
    # ensure the quality of output image
    u0 = image
    m, n, c = u0.shape
    # u0 = u0/255
    lambd = 1/weight
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
    # one mask
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
    # Main loop

    """To display the denoising process, the intermediate values
    are saved to a temp.npy file. """
    # val_lst = [tv_norm(u0),]
    val_lst = [signaltonoise(u0),]
    # store the SNR of images for iteration control
    # iterate for a set number of times
    for i in range(N):
        x_next[:,:,0] = x_solver(z[:,:,0,:], u[:,:,0,:], u0_R, rho, deno)
        z_next[:,:,0,:] = z_solver(x_next[:,:,0], u[:,:,0,:], lambd, rho, mask, tv_type)
        x_next[:,:,1] = x_solver(z[:,:,1,:], u[:,:,1,:], u0_G, rho, deno)
        z_next[:,:,1,:] = z_solver(x_next[:,:,1], u[:,:,1,:], lambd, rho, mask, tv_type)
        x_next[:,:,2] = x_solver(z[:,:,2,:], u[:,:,2,:], u0_B, rho, deno)
        z_next[:,:,2,:] = z_solver(x_next[:,:,2], u[:,:,2,:], lambd, rho, mask, tv_type)

        u_next = u + rho * (D(x_next) - z_next)
        s = -rho * (Div(z_next - z))
        r = D(x) - z
        s_norm = np.linalg.norm(s)
        r_norm = np.linalg.norm(r)
        if r_norm > mu * s_norm:
            rho = rho * tau
        elif s_norm > mu * r_norm:
            rho = rho / tau
        print("ite",i)
        val_temp = float(signaltonoise(x_next))
        if i>1 and val_temp-val_lst[-1]<eps:
            # first time iteration is mostly imaginary number
            # break when noise ratio stops decreasing
            break
        val_lst.append(val_temp)

        x = x_next
        z = z_next
        u = u_next

        x_temp = x.astype(np.uint8)
        os_dir = [str(i),"temp.npy"]
        os_dir = "_".join(os_dir)
        np.save(os_dir, x_temp)
        
    return x, i


"""implements ADMM for TV-L^2."""
def ADMM_2D(image,weight, N, \
              tv_type = "anisotropic",
                rho = 1, mu = 10, tau = 2, ground_truth = None, eps = 1e-10, channel_axis = None):
    # note: when ground truth is nonzero, eps is used to
    # ensure the quality of output image
    """
    image: grayscale noisy image, type m*n np.array
    weight: weight parameter between value 10-20.
    tv_type: ="isotropic" or "anisotropic".
    N: max iter.
    eps: stop iteration when SNR_{i}-SNR_{i-1}<tol.
    """
    lambd= 1/weight
    u0 = image/255
    # normalize input
    x = np.zeros_like(u0)
    # Initialization
    z = D(u0)
    u = np.zeros_like(z)
    mask = np.ones_like(D(u0))
    mask[-1, :, 0] = 0
    mask[:, -1, 1] = 0
    m, n = u0.shape
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
    # tic = time.perf_counter()
    # denoise_process = []
    # Main loop
    # np.save("0_temp.npy",u0)
    """To display the denoising process, the intermediate values
    are saved to temp.npy file. """
    print(signaltonoise(u0))
    snr_lst = [float(signaltonoise(u0)),]
    for k in range(N):
        x_next = x_solver(z, u, u0, rho, deno)
        z_next = z_solver(x_next, u, lambd, rho, mask, tv_type)
        u_next = u + rho * (D(x_next) - z_next)
        
        s = -rho * (Div(z_next - z))
        r = D(x) - z
        s_norm = np.linalg.norm(s)
        r_norm = np.linalg.norm(r)
        if r_norm > mu * s_norm:
            rho = rho * tau
        elif s_norm > mu * r_norm:
            rho = rho / tau
        
        x = x_next
        z = z_next
        u = u_next
        x_temp = x*255
        x_temp = x_temp.astype(np.uint8)
        os_dir = [str(k),"temp.npy"]
        os_dir = "_".join(os_dir)
        np.save(os_dir, x_temp)
        if k>1:
            # first time fft contains imaginary number
            # therefore skip noise comparison when k<=1
            snr = signaltonoise(x_next*255)
            print(snr)
            if snr-snr_lst[-1]<eps:
                break
            snr_lst.append(snr)
        # denoise_process.append(u)
 
    return x*255, k

def ADMM(u0, weight, N, \
              isotropic=True, channel_axis = None,
                rho = 1, mu = 10, tau = 2, ground_truth = None, eps = 1e-6):
    """generalized ADMM for color image."""
    k = 0
    if channel_axis == True: # color image
        assert len(u0.shape)==3, "dimension mismatch"
        # color image
        if isotropic == True:
            tv_type = "isotropic"
        else:
            tv_type = "anisotropic"
        out, k = ADMM_3D(u0,weight, N, \
              tv_type = tv_type,
                rho = rho, mu = mu, tau = tau, ground_truth = None, eps = eps, channel_axis = None)

    else:
        assert len(u0.shape)==2, "dimension mismatch"
        if isotropic == True:
            tv_type = "isotropic"
        else:
            tv_type = "anisotropic"
        out, k = ADMM_2D(u0,weight, N, \
              tv_type = tv_type,
                rho = rho, mu = mu, tau = tau, ground_truth = None, eps = eps, channel_axis = None)
    return out, k
