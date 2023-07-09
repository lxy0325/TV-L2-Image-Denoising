from math import ceil
from math import sqrt
import numpy as np
import os
import cv2
from utils.utils import delete_row_lil, delete_col_lil,delete_row_csr,\
    DiffOper, signaltonoise, image_broadcast
import pdb, scipy, scipy.sparse as sp, scipy.sparse.linalg as splinalg


"""
adaptation from matlab:
https://de.mathworks.com/matlabcentral/fileexchange/36278-split-bregman-method-for-total-variation-denoising
into python3 to include color images.
Algorithm from:
Goldstein and Osher, The split Bregman method for L1 regularized problems
SIAM Journal on Imaging Sciences 2(2) 2009
"""


"""changed iteration stopping scheme to decrease of SNR."""
def SB_ITV(g, mu,rows, cols, N = 100,tol = 1e-10):
    """
    g: noisy image, type 2d np.array
    g should be reshaped to square before input.
    mu: weight parameter between value 10-20.
    rows, cols: for reshaping g after flattening.
    tv_type: = "isotropic" or "anisotropic".
    N: max iter.
    tol: stop iteration when SNR_{i}-SNR_{i-1}<tol.
    """
    g = g.flatten('F')
    n = len(g)
    print(np.sqrt(n))
    broadcast_shape = int(np.sqrt(n))
    B, Bt, BtB = DiffOper(int(np.sqrt(n)))
    b = np.zeros((2*n,1))
    d = b
    u = g
    err = 1
    # k = 1
    # tol = 0.001
    lambda1 = 1 # weight between 2 part of objective function
    
    snr_lst = [float(signaltonoise(g.reshape(broadcast_shape, broadcast_shape))),]
    for k in range(N):
        print ('it. %g '% k)
        up = u
        u,_=sp.linalg.cg(sp.eye(n)+BtB,g-np.squeeze(lambda1*Bt.dot(b-d)),tol=1e-5, maxiter=100)
        Bub = B.dot(u) + np.squeeze(b)
        s = np.sqrt(Bub[:n]**2 + Bub[n:]**2)
        if s[0]==0.:
            s[0]=1.
        d = np.concatenate((np.maximum(s-mu/lambda1,0.)*Bub[:n]/s,np.maximum(s-mu/lambda1,0.)*Bub[n:]/s))
        b = Bub - d
        # err = np.linalg.norm(up - u) / np.linalg.norm(u)
        # print ('err=%g \n'% err)
        # save obtained denoising process
        
        # x_temp = x_temp.reshape((rows,cols), order = "F")
        x_temp = u.reshape((broadcast_shape,broadcast_shape),order = "F")
        snr = float(signaltonoise(x_temp))
        print(snr)
        if snr-snr_lst[-1]<tol:
            break
        snr_lst.append(snr)
        x_temp = x_temp.astype(np.uint8)
        x_temp = cv2.resize(x_temp, (rows, cols), interpolation = cv2.INTER_AREA)
        # x_temp = x_temp[int((broadcast_shape-rows)/2):int(broadcast_shape-(broadcast_shape-rows)/2), \
        #        int((broadcast_shape-cols)/2):int(broadcast_shape-(broadcast_shape-cols)/2)]
        os_dir = [str(k),"temp.npy"]
        os_dir = "_".join(os_dir)
        # for color image: if existent then append to file
        if os.path.exists(os_dir):
            x_existent = np.load(os_dir)
            _,_,c = x_existent.shape
            lst = [x_existent[:,:,i] for i in range(c)]
            lst.append(x_temp)
            lst=np.array(lst)
            np.save(os_dir, lst)
        else:
            np.save(os_dir, x_temp)
    
    #if err<eps:
    #    break
    
    print ("Stopped because SNR stops increasing.")  #norm(up-u)/norm(u) <= tol=%.1e\n'% tol)
    return u, k

"""changed iteration stopping scheme to decrease of SNR."""
def SB_ATV(g, mu,rows, cols, N = 100,tol = 1e-6):
    """
    g: noisy image, type 2d np.array
    g should be reshaped to square before input.
    mu: weight parameter between value 10-20.
    rows, cols: for reshaping g after flattening.
    tv_type: = "isotropic" or "anisotropic".
    N: max iter.
    tol: stop iteration when SNR_{i}-SNR_{i-1}<tol.
    """
    g = g.flatten()
    n = len(g)
    print(np.sqrt(n))
    broadcast_shape = int(np.sqrt(n))
    B, Bt, BtB = DiffOper(int(np.sqrt(n)))
    b = np.zeros((2 * n, 1))
    d = b
    u = g
    k = 1
    lambda1 = 1 # weight between 2 parts of objective function
     
    snr_lst = [float(signaltonoise(g.reshape(broadcast_shape, broadcast_shape))),]
    for k in range(N):
        print ('it. %d ' % k)
        up = u
        u, _ = splinalg.cg(sp.eye(n) + BtB, g - np.squeeze(lambda1 * Bt.dot(b - d)), tol=1e-05, maxiter=100)
        Bub = B * u + np.squeeze(b)
        print (np.linalg.norm(Bub))
        d = np.maximum(np.abs(Bub) - mu / lambda1,0) * np.sign(Bub)
        b = Bub - d
        # err = np.linalg.norm(up - u) / np.linalg.norm(u)
        # print ('err=%g'%err)

        x_temp = u.astype(np.uint8)
        x_temp = x_temp.reshape((broadcast_shape,broadcast_shape),order = "F")
        snr = signaltonoise(x_temp)
        print(snr)
        if snr-snr_lst[-1]<tol:
            break
        snr_lst.append(snr)
        # shape squared image to original size
        x_temp = cv2.resize(x_temp, (cols, rows), interpolation = cv2.INTER_AREA)
        
        os_dir = [str(k),"temp.npy"]
        os_dir = "_".join(os_dir)
        # for color image: if existent then append to file
        if os.path.exists(os_dir):
            x_existent = np.load(os_dir)
            _,_,c = x_existent.shape
            lst = [x_existent[:,:,i] for i in range(c)]
            lst.append(x_temp)
            lst=np.array(lst)
            np.save(os_dir, lst)
        else:
            np.save(os_dir, x_temp)
        # np.save(os_dir, x_temp)
        # save obtained denoising process
    # output is vector, should be reshaped/ truncated.
    print ("Stopped because SNR stops increasing.")
    return u, k


def split_bregman(u0, weight, N, original_size, isotropic=True, channel_axis = None, eps = 1e-6):
    """
    to include color image. By concatenating color channels
     that were separately denoised, one may obtain a denoised color image.
        we define param. weight as 1/lambd which represents the weight of 
          image regularization.  
    u0 : noisy input as array.
    weight: weight parameter 1/lambda usually between 10-20.
    N: maxiter.
    original_size: for reshaping inputs after flattening. list or tuple.
    """
    
    # first broadcast the image to square
    rows = original_size[0]
    cols = original_size[1]
    out = np.zeros_like(u0)
    # u0 = image_broadcast(u0)
    broadcast_shape = u0.shape[0]
    if channel_axis == True: # color image
        # TODO: color image iteration 
        s = u0.shape
        rows = s[0]
        cols = s[1]
        dims = s[-1]
        u0 = image_broadcast(u0)
        
        for i in range(dims):
            f = np.reshape(u0[:,:,i], (broadcast_shape, broadcast_shape), order = "F")
            g = f.flatten("F") # column-major flatten
            if isotropic == True:
                g_denoise, k = SB_ITV(g, weight, rows, cols, tol = eps, N = N)
            else:
                g_denoise, k = SB_ATV(g, weight, rows, cols, tol = eps, N = N)
            g_denoise = g_denoise.reshape((broadcast_shape,broadcast_shape),order = "F")
            g_denoise = g_denoise[int((broadcast_shape-rows)/2):int(broadcast_shape-(broadcast_shape-rows)/2), \
                                  int((broadcast_shape-cols)/2):int(broadcast_shape-(broadcast_shape-cols)/2)]
            out[:,:,i] = g_denoise
    else:
        s = u0.shape
        rows = s[0]
        cols = s[1]
        u0 = image_broadcast(u0)
        out = np.zeros_like(u0)
        g = u0.flatten('F')
        if isotropic == True:
            out, k= SB_ITV(g, weight, rows, cols, tol = eps, N = N)
        else:
            out, k = SB_ATV(g, weight, rows, cols, tol = eps, N = N)
        
        out = out.reshape((broadcast_shape,broadcast_shape),order = "F")
        # out = out[int((broadcast_shape-rows)/2):int(broadcast_shape-(broadcast_shape-rows)/2), \
        # int((broadcast_shape-cols)/2):int(broadcast_shape-(broadcast_shape-cols)/2)]
        out = cv2.resize(out, (cols, rows), interpolation = cv2.INTER_AREA)
    return out, k

