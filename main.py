import numpy as np
import f

from time import time
from tqdm import tqdm
from skimage.io import imread, imsave


if __name__ == '__main__':

    img = imread("data/map_hsv.png")
    # img = np.ones((4, 4))
    width, height, t_max = img.shape[0], img.shape[1], img.shape[0] * img.shape[1]
    img = img.reshape((t_max, 3))

    # Init
    Res = np.ones(t_max, dtype=np.uint8)
    Phi = np.zeros((t_max, 4, 2), dtype=np.float)
    g = f.init_g(1)
    q = f.init_q(img)
    N = f.init_neighbours(width, height)
    K = f.init_k(img)

    for iter in range(0, 10):

        print(f"[ {iter+1} | 5 ]")
        K_next, Res = f.iteration(N, K, Phi, Res, g, q, t_max)
        imsave(f"data/results/res{iter+1}.png", ( Res.reshape((width, height)))*255 )
        K = K_next

    # imsave(f"data/results/test.png", K1.reshape((width, height)) - K.reshape((width, height)))
    print(f"K1 shape: {Res.shape} | min, max: {np.min(Res), np.max(Res)} | dtype: {Res.dtype}")





