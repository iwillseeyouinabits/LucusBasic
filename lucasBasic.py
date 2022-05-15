import numpy as np
from scipy import signal
import cv2
import numpy as np


def optical_flow(I1path, I2path, window_size, tau=1e-2):
    I1 = cv2.imread(I1path)
    I2 = cv2.imread(I2path)
    I1g = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2g = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    kernel_x = np.array([[-1., 1.], [-1., 1.]])
    kernel_y = np.array([[-1., -1.], [1., 1.]])
    kernel_t = np.array([[1., 1.], [1., 1.]])  # *.25
    w = window_size // 2  # window_size is odd, all the pixels with offset in between [-w, w] are inside the window
    I1g = I1g / 255.  # normalize pixels
    I2g = I2g / 255.  # normalize pixels
    # Implement Lucas Kanade
    # for each point, calculate I_x, I_y, I_t
    mode = 'same'
    fx = signal.convolve2d(I1g, kernel_x, boundary='symm', mode=mode)
    fy = signal.convolve2d(I1g, kernel_y, boundary='symm', mode=mode)
    ft = signal.convolve2d(I2g, kernel_t, boundary='symm', mode=mode) + signal.convolve2d(I1g, -kernel_t, boundary='symm', mode=mode)

    u = np.zeros(I1g.shape)
    v = np.zeros(I1g.shape)
    # within window window_size * window_size
    for i in range(w, I1g.shape[0] - w):
        for j in range(w, I1g.shape[1] - w):
            Ix = fx[i - w:i + w + 1, j - w:j + w + 1].flatten()
            Iy = fy[i - w:i + w + 1, j - w:j + w + 1].flatten()
            It = ft[i - w:i + w + 1, j - w:j + w + 1].flatten()
            A = np.concatenate(([Ix],[Iy]))
            AT = np.transpose(A)
            aInvTran = np.transpose(np.dot(A,AT))
            nu = np.dot(aInvTran, np.dot(It,AT))

            # if threshold τ is larger than the smallest eigenvalue of A'A:
            u[i, j] = nu[0]
            v[i, j] = nu[1]

    return (u, v)
def createFlowMap(disp,h,w,window):
    max,min = maxMinDisp(disp)

    print(max)
    img = [[[0, 0, 0] for i in range(w)] for j in range(h)]
    for i in range(h):
        for j in range(w):
            val = abs(disp[0][i][j])+abs(disp[1][i][j])
            valScaled = ((val-min)/(max-min))*(255)
            img[i][j] =[valScaled,valScaled,valScaled]
    cv2.imwrite('dispTest2.png', np.array(img))
def maxMinDisp(of):
    max = -1000000000
    min = 100000000
    for i in range(len(of[0])):
        for j in range(len(of[0][0])):
            disp = abs(of[0][i][j])+abs(of[0][i][j])
            if max<=disp:
                max = disp
            if min>=disp:
                min = disp
    return max,min
def main():
    of = optical_flow("1110lhs.png", "1110.png", 6)
    measure = cv2.imread("test1.png")
    print(measure.shape[0])
    createFlowMap(of,measure.shape[0],measure.shape[1],6)
    np.set_printoptions(threshold=np.inf)

main()