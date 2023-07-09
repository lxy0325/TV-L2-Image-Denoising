import numpy as np
import cv2 
from utils.utils import deltaX_p,deltaX_n,\
    deltaY_p,deltaY_n,minmod,_lambda

"""
The code realizes the gradient descent algorithm as orginally proposed in ROF paper:
Rudin, L. I.; Osher, S.; Fatemi, E. (1992). "Nonlinear total variation based noise removal algorithms".
The input is normalized to a value between 0 and 1 before processing.
Due to normalization the eps should be also adjusted.
It takes approx. 5 min to run since multiple for loops are involved. 
We do not recommend anyone to run this for the purpose of image denoising.
"""

def ROFtv(u0, N=300, weight=10, deltaT=1e-6, show=False, eps = 1e-6):
    """
    u0: grayscale noisy image, type m*n np.array
    weight: weight parameter between value 10-20.
    N: max iter.
    eps: stop iteration when np.norm(u_{i}-u_{i-1})<tol.
    """
    sigma = 1/weight

    u0 = u0/255.0 # normalization to 0-1
    l = 0
    normValue = 0
    s = u0.shape
    h = 1 / s[1]
    U0x = np.zeros_like(u0, dtype=np.float32)
    U0y = np.zeros_like(u0, dtype=np.float32)
    Ux = np.zeros_like(u0, dtype=np.float32)
    Uy = np.zeros_like(u0, dtype=np.float32)
    xU = np.zeros_like(u0, dtype=np.float32)
    yU = np.zeros_like(u0, dtype=np.float32)
    pool = np.zeros_like(u0, dtype=np.float32)
    X = np.zeros_like(u0, dtype=np.float32)
    Y = np.zeros_like(u0, dtype=np.float32)
    u = u0.astype(np.float32)
    u_last = np.zeros_like(u)
    # u = np.zeros_like(u0, dtype=np.float32)

    for k in range(N):
        print(k)
        Ux = deltaX_p(u)
        xU = deltaX_n(u)
        Uy = deltaY_p(u)
        yU = deltaY_n(u)

        if k == 0:
            U0x = Ux.copy()
            U0y = Uy.copy()

        if k > 0:
            l = _lambda(U0x, U0y, Ux, Uy, -0.5 * h / sigma)

        for i in range(s[0]):
            for j in range(s[1]):
                X[i, j] = Ux[i, j] / (np.sqrt(Ux[i, j] * Ux[i, j] + minmod(yU[i, j], Uy[i, j]) ** 2) + 0.00001) / h
                Y[i, j] = Uy[i, j] / (np.sqrt(Uy[i, j] * Uy[i, j] + minmod(xU[i, j], Ux[i, j]) ** 2) + 0.00001) / h

        pool = u - u0
        temp = -l * pool
        pool = deltaX_n(X)
        temp += pool
        pool = deltaY_n(Y)
        temp += pool
        temp *= deltaT
        normValue = np.sum(np.abs(temp[1:-1, 1:])) * (h ** 2)
        u_last = u[:,:]
        u += temp

        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        print("i = ", k)        
        if np.linalg.norm(255*(u_last-u))/np.linalg.norm(255*u)<eps:
            print(np.linalg.norm(255*(u_last-u))/np.linalg.norm(255*u))
            break

        if show:
            cv2.putText(u, f"k={k}   |u-u0|={normValue}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255))
            cv2.imshow("ing", u)
            cv2.waitKey(1)
        x_temp = (u*255).astype(np.uint8)
        os_dir = [str(k),"temp.npy"]
        os_dir = "_".join(os_dir)
        np.save(os_dir, x_temp)

    u = u*255
    return u, k