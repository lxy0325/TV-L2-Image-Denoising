# import cv2
import numpy as np

def deltaX_p(u):
    result = np.zeros_like(u)
    for i in range(u.shape[0]):
        for j in range(u.shape[1] - 1):
            result[i, j] = -u[i, j] + u[i, j + 1]
    return result

def deltaX_n(u):
    result = np.zeros_like(u)
    for i in range(u.shape[0]):
        result[i, 0] = 0
        for j in range(1, u.shape[1]):
            result[i, j] = -u[i, j - 1] + u[i, j]
    return result

def deltaY_p(u):
    result = np.zeros_like(u)
    for j in range(u.shape[1]):
        for i in range(u.shape[0] - 1):
            result[i, j] = u[i + 1, j] - u[i, j]
    return result

def deltaY_n(u):
    result = np.zeros_like(u)
    for j in range(u.shape[1]):
        result[0, j] = 0
        for i in range(1, u.shape[0]):
            result[i, j] = -u[i - 1, j] + u[i, j]
    return result

def minmod(a, b):
    # a, b are 2 scalar values
    sgn_a = np.sign(a)
    sgn_b = np.sign(b)
    return 0.5 * (sgn_a + sgn_b) * min(abs(a), abs(b))

def _lambda(u0x, u0y, ux, uy, coeff):
    s = u0x.shape
    lam = 0
    for i in range(s[0]):
        for j in range(s[1]):
            temp = ux[i, j] ** 2 + uy[i, j] ** 2
            elem = temp - ux[i, j] * u0x[i, j] - uy[i, j] * u0y[i, j]
            elem /= (np.sqrt(temp) + 0.0001)
            lam += elem
    lam *= coeff
    return lam
