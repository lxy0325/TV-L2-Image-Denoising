import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2
import time
from utils.utils import D, DT

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
    x = ifft2(fft2(u0 + rho * DT(z) - DT(u)) / deno)
    return x

"""implements ADMM for TV-L^2."""
def ADMM_2D(u0,lambd, N, \
              tv_type = "anisotropic",
                rho = 1, mu = 10, tau = 2, ground_truth = None, eps = 1e-3, channel_axis = None):
    # note: when ground truth is nonzero, eps is used to
    # ensure the quality of output image
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
    are saved to a temp.npy file. """
    # TODO: find RAM-economical solution to dynamical denoising display
    if ground_truth == None:
        # iterate for a set number of times
        for i in range(N):
            # TODO: set stopping criteria with original image
            x_next = x_solver(z, u, u0, rho, deno)
            z_next = z_solver(x_next, u, lambd, rho, mask, tv_type)
            
            u_next = u + rho * (D(x_next) - z_next)
            
            s = -rho * (DT(z_next - z))
            r = D(x) - z
            s_norm = np.linalg.norm(s)
            r_norm = np.linalg.norm(r)
            if r_norm > mu * s_norm:
                rho = rho * tau
            elif s_norm > mu * r_norm:
                rho = rho / tau
            if np.linalg.norm(x-x_next)/np.linalg.norm(x) < eps:
                break
            x = x_next
            z = z_next
            u = u_next
            x_temp = x.astype(np.uint8)
            os_dir = [str(i),"temp.npy"]
            os_dir = "_".join(os_dir)
            np.save(os_dir, x_temp)
            # denoise_process.append(u)
    else:
        # compare the generated image with ground truth
        # end iteration if difference smaller than eps
        # or when k exceeds maxiter
        k = 0
        while np.linalg.norm(ground_truth, x)/np.linalg.norm(x)>eps:
            x_next = x_solver(z, u, u0, rho, deno)
            z_next = z_solver(x_next, u, lambd, rho, mask, tv_type)
            u_next = u + rho * (D(x_next) - z_next)
            
            s = -rho * (DT(z_next - z))
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
            x_temp = x.astype(np.uint8)
            os_dir = [str(k),"temp.npy"]
            os_dir = "_".join(os_dir)
            np.save(os_dir, x_temp)
            # denoise_process.append(u)
            if k > N:
                break
            k+=1
    # toc = time.perf_counter()
    # runtime = toc - tic
    
    return x

def ADMM(u0, lambd, N, \
              isotropic=True, channel_axis = None,
                rho = 1, mu = 10, tau = 2, ground_truth = None, eps = 1e-3):
    """generalized ADMM for color image."""
    if channel_axis == True: # color image
        assert len(u0.shape)==3, "dimension mismatch"
        # color image
        s = u0.shape
        rows = s[0]
        cols = s[1]
        dim = s[-1]
        out = np.zeros_like(u0)
        # out_list = []
        for i in range(dim):
            f = np.reshape(u0[:,:,i], (rows, cols))
            if isotropic == True:
                tv_type = "isotropic"
            else:
                tv_type = "anisotropic"
            f_denoise = ADMM_2D(f,lambd, N, \
              tv_type = tv_type,
                rho = rho, mu = mu, tau = tau, \
                    ground_truth = None, eps = eps, channel_axis = True)#, channel_axis = True)
            out[:,:,i] = f_denoise
            # TODO: color image denoise process save file

    else:
        assert len(u0.shape)==2, "dimension mismatch"
        if isotropic == True:
            tv_type = "isotropic"
        else:
            tv_type = "anisotropic"
        out = ADMM_2D(u0,lambd, N, \
              tv_type = tv_type,
                rho = rho, mu = mu, tau = tau, ground_truth = None, eps = eps, channel_axis = None)
    return out

