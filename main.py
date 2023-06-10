import numpy as np
import cv2 
from finite_difference import deltaX_p,deltaX_n,\
    deltaY_p,deltaY_n,minmod,_lambda

def ROFtv(u0, N=300, sigma=0.002, deltaT=1e-6, show=False):
    index = 0
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

        u += temp

        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]

        if show:
            cv2.putText(u, f"k={k}   |u-u0|={normValue}", (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255))
            cv2.imshow("ing", u)
            cv2.waitKey(1)

    return u


if __name__ == "__main__":
    fileName = "input.png"
    N = 300
    sigma2 = 0.002
    deltaT = 1e-6
    show = False

    u = cv2.imread("E:\\2nd-semester\practical\\test_v1\input.png", cv2.IMREAD_GRAYSCALE)
    print(u)
    u = u.astype(np.float32) / 255.0

    result = ROFtv(u, N, sigma2, deltaT, show)
    result *= 255
    result = result.astype(np.uint8)
    cv2.imwrite("E:\\2nd-semester\practical\\test_v1\output.png", result)
