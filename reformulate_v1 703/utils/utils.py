import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2
import time
import pdb, scipy, scipy.sparse as sp, scipy.sparse.linalg as splinalg
# Helper functions for finite differences

def D(x):
    # forward difference
    s = x.shape
    if len(s)==2: 
    #grayscale image
        grad = np.zeros([s[0], s[1],2])
        grad[:,:,0] = x - np.roll(x, (-1, 0), axis=(0, 1))
        grad[s[0]-1, :, 0] = 0
        grad[:, :, 1] = x - np.roll(x, (0, -1), axis=(0, 1))
        grad[:, s[1]-1, 1] = 0
        return grad
    else:
        # color image
        assert len(x.shape)==3, "image dimension error: dimension not accepted."
        n1, n2, _ = x.shape
        grad = np.zeros((n1, n2, 3, 2))

        r = x[:, :, 0].squeeze()
        g = x[:, :, 1].squeeze()
        b = x[:, :, 2].squeeze()

        # TODO: optimize code structure & reduce repetition

        grad[:, :, 0, 0] = r - np.roll(r, shift=(-1, 0), axis=(0, 1))
        grad[n1 - 1, :, 0, 0] = 0
        grad[:, :, 0, 1] = r - np.roll(r, shift=(0, -1), axis=(0, 1))
        grad[:, n2 - 1, 0, 1] = 0

        grad[:, :, 1, 0] = g - np.roll(g, shift=(-1, 0), axis=(0, 1))
        grad[n1 - 1, :, 1, 0] = 0
        grad[:, :, 1, 1] = g - np.roll(g, shift=(0, -1), axis=(0, 1))
        grad[:, n2 - 1, 1, 1] = 0

        grad[:, :, 2, 0] = b - np.roll(b, shift=(-1, 0), axis=(0, 1))
        grad[n1 - 1, :, 2, 0] = 0
        grad[:, :, 2, 1] = b - np.roll(b, shift=(0, -1), axis=(0, 1))
        grad[:, n2 - 1, 2, 1] = 0
        
        return grad

# calculates div
def DT(grad):
    if len(grad.shape)==3:
        # grayscale image
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
    else:
        # color image
        assert len(grad.shape)==4, "gradient dimension error: dimension not accepted."
        n1, n2, _, _ = grad.shape
        div = np.zeros((n1, n2, 3))

        for k in range(3):
            for i in range(2):
                shift = np.roll(grad[:, :, k, i].squeeze(), shift=(1-i, -i), axis=(0, 1))
                div1 = grad[:, :, k, i].squeeze() - shift
                div1[0, :] = grad[0, :, k, i].squeeze()
                div1[n1 - 1, :] = -shift[n1 - 1, :]

                shift = np.roll(grad[:, :, k, 1-i].squeeze(), shift=(-i, 1-i), axis=(0, 1))
                div2 = grad[:, :, k, 1-i].squeeze() - shift
                div2[:, 0] = grad[:, 0, k, 1-i].squeeze()
                div2[:, n2 - 1] = -shift[:, n2 - 1]

                div[:, :, k] += div1 + div2
        return div

def delete_row_lil(mat, i):
    if not isinstance(mat, scipy.sparse.lil_matrix):
        raise ValueError('works only for LIL format -- use .tolil() first')
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])


def delete_col_lil(mat, i):
    if not isinstance(mat, scipy.sparse.lil_matrix):
        raise ValueError('works only for LIL format -- use .tolil() first')
    mat.cols = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0], mat._shape[1] - 1)


def delete_row_csr(mat, i):
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError('works only for CSR format -- use .tocsr() first')
    n = mat.indptr[i + 1] - mat.indptr[i]
    if n > 0:
        mat.data[(mat.indptr[i]):(-n)] = mat.data[mat.indptr[i + 1]:]
        mat.data = mat.data[:-n]
        mat.indices[(mat.indptr[i]):(-n)] = mat.indices[mat.indptr[i + 1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:(-1)] = mat.indptr[i + 1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0] - 1, mat._shape[1])


def DiffOper(N):
    data = np.vstack([-np.ones((1, N)), np.ones((1, N))])
    D = sp.diags(data, [0, 1], (N, N + 1), 'csr')
    #print 'shape before: ', D.shape
    D = D[:, 1:]
    #print 'shape afterward: ', D.shape
    D[(0, 0)] = 0
    #print 'D dimensions: ', D.shape
    B = sp.vstack([sp.kron(sp.eye(N), D), sp.kron(D, sp.eye(N))], 'csr')
    Bt = B.transpose().tocsr()
    BtB = Bt * B
    #print 'BtB dimensions: ', BtB.shape
    #print 'Returned'
    return B, Bt, BtB