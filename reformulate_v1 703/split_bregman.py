from math import ceil
from math import sqrt
import numpy as np
import os
import cv2
from utils.utils import delete_row_lil, delete_col_lil,delete_row_csr,DiffOper
import pdb, scipy, scipy.sparse as sp, scipy.sparse.linalg as splinalg


"""
adaptation from:
https://github.com/melrobin/SplitBregman
into python3 to include color image.
"""



def SB_ITV(g, mu,rows, cols, N = 100,tol = 1e-3):
     
     g = g.flatten('F')
     n = len(g)
     B, Bt, BtB = DiffOper(int(np.sqrt(n)))
     b = np.zeros((2*n,1))
     d = b
     u = g
     err = 1
     k = 1
     # tol = 0.001
     lambda1 = 1
     while err > tol:
        print ('it. %g '% k)
        up = u
        u,_=sp.linalg.cg(sp.eye(n)+BtB,g-np.squeeze(lambda1*Bt.dot(b-d)),tol=1e-5, maxiter=100)
        Bub = B.dot(u) + np.squeeze(b)
        s = np.sqrt(Bub[:n]**2 + Bub[n:]**2)
        if s[0]==0.:
            s[0]=1.
        d = np.concatenate((np.maximum(s-mu/lambda1,0.)*Bub[:n]/s,np.maximum(s-mu/lambda1,0.)*Bub[n:]/s))
        b = Bub - d
        err = np.linalg.norm(up - u) / np.linalg.norm(u)
        print ('err=%g \n'% err)
        # save obtained denoising process
        x_temp = u.astype(np.uint8)
        x_temp = x_temp.reshape((rows,cols), order = "F")
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
        k = k + 1
        if k>N:
            break
     print ('Stopped because norm(up-u)/norm(u) <= tol=%.1e\n'% tol)
     return u

def SB_ATV(g, mu,rows, cols, N = 100,tol = 1e-3):
    g = g.flatten()
    n = len(g)
    B, Bt, BtB = DiffOper(int(np.sqrt(n)))
    b = np.zeros((2 * n, 1))
    d = b
    u = g
    err = 1
    k = 1
    lambda1 = 1
    while err > tol:
        print ('it. %d ' % k)
        up = u
        u, _ = splinalg.cg(sp.eye(n) + BtB, g - np.squeeze(lambda1 * Bt.dot(b - d)), tol=1e-05, maxiter=100)
        Bub = B * u + np.squeeze(b)
        print (np.linalg.norm(Bub))
        d = np.maximum(np.abs(Bub) - mu / lambda1,0) * np.sign(Bub)
        b = Bub - d
        err = np.linalg.norm(up - u) / np.linalg.norm(u)
        print ('err=%g'%err)
        x_temp = u.astype(np.uint8)
        x_temp = x_temp.reshape((rows,cols),order = "F")
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
        k = k + 1
        if k>N:
            break
        # save obtained denoising process

    print ('Stopped because norm(up-u)/norm(u) <= tol=%.1e\n'% tol)
    return u

def split_bregman(u0, weight, N, isotropic=True, channel_axis = None, eps = 1e-3):
    """to include color image. By concatenating color channels
     that were separately denoised, one may obtain a denoised color image.
      One may alsp solve for the vectorized version of ROF, with 
       an extra dimension for color channels.
        we define param. weight as 1/lambd which represents the weight of 
          image regularization.  """
    
    if channel_axis == True: # color image
        # TODO: assert dim, error message
        s = u0.shape
        rows = s[0]
        cols = s[1]
        dims = s[-1]
        out = np.zeros_like(u0)
        for i in range(dims):
            f = np.reshape(u0[:,:,i], (rows, cols), order = "F")
            g = f.flatten("F") # column-major flatten
            if isotropic == True:
                g_denoise = SB_ITV(g, weight, rows, cols, tol = eps)
            else:
                g_denoise = SB_ATV(g, weight, rows, cols, tol = eps)
            g_denoise = g_denoise.reshape((rows,cols),order = "F")
            out[:,:,i] = g_denoise
    else:
        s = u0.shape
        rows = s[0]
        cols = s[1]
        out = np.zeros_like(u0)
        g = u0.flatten('F')
        if isotropic == True:
            out = SB_ITV(g, weight, rows, cols, tol = eps)
        else:
            out = SB_ATV(g, weight, rows, cols, tol = eps)
        out = out.reshape((rows,cols),order = "F")
    return out


