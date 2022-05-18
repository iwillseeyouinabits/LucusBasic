import numpy as np
from scipy import signal
import cv2
import numpy as np
import copy


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

def createFlowMap(u,v,flMap):
    disp = abs(u) + abs(v)
    np.save(flMap, disp)
    # np.set_printoptions(threshold=np.inf)
    # np.set_printoptions(suppress=True)
    # print(np.load(flMap))

def createClowMaps(startInd, endInd):
    for i in range(startInd, endInd+1):
        num = str(i)
        file1 = str("0"*(6-len(num))) + num
        path1 = "image_2/" + file1 + ".png"
        path2 = "image_3/" + file1 + ".png"
        print(path1)
        print(path2)
        print("______________")
        u , v = optical_flow(path1, path2, 6)
        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(suppress=True)
        # print(abs(u) + abs(v))
        createFlowMap(u, v, "FlowMap/flowMapV" + str(i) + ".npy")

def compare(disp, groundTruth, k):
    d = np.load(disp)
    gt = np.load(groundTruth)
    num = 0
    den = 0
    for x in range(len(d)):
        for y in range(len(d[x])):
            pxd = d[x][y]
            pxgt = gt[x][y]
            if not pxd <= 10 and not pxgt <= 0:
                # print(str(pxd) + " " + str(pxgt))
                if (pxd - pxgt)**2 > k**2:
                    num += 1
                den += 1
    return num, den

def compareAll(start, end, k):
    num = 0
    den = 0
    for i in range(start, end+1):
        number = str(i)
        file1 = str("0"*(6-len(number))) + number
        gt = "GroundTruth/" + file1 + ".npy"
        disp = "FlowMap/flowMapV" + number + ".npy"
        tempNum, tempDen = compare(disp, gt, k)
        num += tempNum
        den += tempDen
        print(str(i) + "  " + str(num/den))
    print(num/den)
    return num/den


if __name__ == '__main__':
    compareAll(0, 7480, 1)