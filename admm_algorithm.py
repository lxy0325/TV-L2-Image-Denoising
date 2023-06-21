import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2
import time

# Helper functions for finite differences
# calculates the gradients of an image x 
def D(x):
    s = x.shape
    grad = np.zeros([s[0], s[1],2])
    grad[:,:,0] = x - np.roll(x, (-1, 0), axis=(0, 1))
    grad[s[0]-1, :, 0] = 0
    grad[:, :, 1] = x - np.roll(x, (0, -1), axis=(0, 1))
    grad[:, s[1]-1, 1] = 0
    return grad

# calculates the adjoint operator of D
def DT(grad):
    n1, n2, _ = grad.shape
    shift = np.roll(grad[:, :, 0], (1, 0), axis=(0, 1))
    div1 = grad[:, :, 0] - shift
    div1[0, :] = grad[0, :, 0]
    div1[n1-1, :] = -shift[n1-1, :]

    shift = np.roll(grad[:, :, 1], (0, 1), axis=(0, 1))
    div2 = grad[:, :, 1] - shift
    div2[:, 0] = grad[:, 0, 1]
    div2[:, n2-1] = -shift[:, n2-1]

    div = div1 + div2 
    return div

# Auxiliary functions
def z_solver(x, u, lambd, rho, mask, tv_type):
    w = mask * (D(x) + (1 / rho) * u)
    if tv_type == 'anisotropic':
        z = soft_threshold(w, lambd / rho)
    else:
        w_v, w_h = w[:, :, 0], w[:, :, 1]
        t = np.sqrt(w_v**2 + w_h**2)
        z = soft_threshold(t, lambd / rho) / (t + np.finfo(float).eps) * w
    return z

def soft_threshold(x, kappa):
    return np.maximum(x - kappa, 0) - np.maximum(-x - kappa, 0)

def x_solver(z, u, u0, rho, deno):
    x = ifft2(fft2(u0 + rho * DT(z) - DT(u)) / deno)
    return x

"""implements ADMM for TV-L^2."""
def ADMM_gray(u0,lambd, N, \
              tv_type = "anisotropic",
                rho = 1, mu = 10, tau = 2, ground_truth = None, eps = 1e-3):
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
    dv = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
    dv_pad = np.zeros((m, n))
    dv_pad[m//2:m//2+3, n//2:n//2+3] = dv
    fdh = fft2(dh_pad)
    fdv = fft2(dv_pad)
    deno = 1 + rho * np.abs(fdh)**2 + rho * np.abs(fdv)**2
    tic = time.perf_counter()
    # Main loop
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
            
            x = x_next
            z = z_next
            u = u_next
    else:
        # compare the generated image with ground truth
        # end iteration if difference smaller than eps
        # or when k exceeds maxiter
        k = 1
        while np.linalg.norm(ground_truth, x)>eps:
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
            if k > N:
                break
    toc = time.perf_counter()
    runtime = toc - tic
    
    return x

if __name__ == "__main__":
    fileName = "input.png"
    N = 100
    weight = 40

    u = cv2.imread("input.png", cv2.IMREAD_GRAYSCALE)
    #print(u)
    u = u.astype(np.float32)#/ 255.0

    result, runtime = ADMM_gray(u,weight, N)
    #result *= 255
    result = result.astype(np.uint8)
    cv2.imwrite("output_ADMM.png", result)
