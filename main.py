import numpy as np
import f

from time import time
from skimage.io import imread, imsave


if __name__ == '__main__':

    img = imread("data/map_hsv.png")
    imsave(f"data/results/test.png", img[:,:,1])
    # img = np.ones((4, 4))
    width, height, t_max = img.shape[0], img.shape[1], img.shape[0] * img.shape[1]
    img = img.reshape((t_max, 3))

    # Init
    Res = np.ones(t_max)
    Phi = np.zeros((t_max, 4, 2))
    g = f.init_g(1)
    q = f.init_q(img)
    N = f.init_neighbours(width, height)

    for iter in range(0, 10):
        print(f"iter {iter}")
        Phi = f.iteration(N, Phi, g, q, t_max)
        

    Img = f.reconstruction(Phi, N, g, q, Res, t_max)
    imsave(f"data/results/res{iter+1}.png", ( Img.reshape((width, height)))*255 )
    print(f"K1 shape: {Img.shape} | min, max: {np.min(Img), np.max(Img)} | dtype: {Img.dtype}")
