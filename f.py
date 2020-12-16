import numpy as np

from numba import njit


def init_neighbours(width, height):

    N = np.full( (width*height, 4), -1)

    borders = get_borders_ind(width, height)
    for t in range(0, width * height):

        # corners
        if t == 0:
            N[t] = [t+1, -1, t+width, -1]
        elif t == width-1:
            N[t] = [-1, t-1, t+width, -1]
        elif t == (height - 1) * width:
            N[t] = [t+1, -1, -1, t-width]
        elif t == width * height - 1:
            N[t] = [-1, t-1, -1, t-width]

        # borders
        elif t in borders[0]:  # top
            N[t] = [t+1, t-1, t+width, -1]

        elif t in borders[1]:  # bottom
            N[t] = [t+1, t-1, -1, t-width]

        elif t in borders[2]:  # right
            N[t] = [-1, t-1, t+width, t-width]

        elif t in borders[3]:  # left
            N[t] = [t+1, -1, t+width, t-width]

        # inside elements
        else:
            N[t] = [t+1, t-1, t+width, t-width]

    return N


def get_borders_ind(w, h):

    borders = [[], [], [], []]                # [top, bottom, right, left]

    for i in range(1, w-1):
        borders[0].append(i)                  # top
        borders[1].append( (h-1)*w + i )      # bottom

    for j in range(1, h-1):
        borders[2].append(2*w-1 + (j-1)*w)  # right
        borders[3].append(w*j)              # left

    return borders


def init_k(img):
    # return np.zeros((img.shape[0], 2), dtype=np.uint8)
    return np.random.randint( 2, size=(img.shape[0], 2), dtype=np.uint8 )


def init_g(alpha):
    return np.array([[alpha, 0], [0, alpha]])


def init_q(img):

    t_max = img.shape[0]
    c = np.array([[0., 255., 0.], [0., 0., 255.]])
    q = np.zeros((t_max, 2))

    for t in range(0, t_max):
        for k in [0,1]:
            if np.linalg.norm( img[t]-c[k] ) > np.linalg.norm( img[t]-c[ abs(k - 1) ] ):
                q[t, k] = 1
            else:
                q[t, k] = 0

    return q


# --------------------------------------------


@njit
def iteration(N, K, Phi, Res, g, q, t_max):

    for t in range(0, t_max):
        for k in [0, 1]:

            # K*
            for t_n in N[t]:
                if t_n != -1:
                    foo = [-1, -1]
                    for k_n in [0, 1]:
                        t_pos = np.where(N[t_n] == t)[0][0]
                        t_posn = np.where(N[t] == t_n)[0][0]
                        foo[k_n] = g[k, k_n] - Phi[t, t_posn, k] - Phi[t_n, t_pos, k_n]

                    K[t_n, k] = foo.index(max(foo))  # t_n?

            # C
            C, nb_amnt = 0, 0
            for t_n in N[t]:
                if t_n != -1:
                    t_pos = np.where(N[t_n] == t)[0][0]
                    C += g[k, K[t_n, k]] - Phi[t_n, t_pos, K[t_n, k]] + q[t, k]
                    nb_amnt += 1
            C = C/nb_amnt

            # Phi
            for t_n in N[t]:
                if t_n != -1:
                    t_posn = np.where(N[t] == t_n)[0][0]
                    t_pos = np.where(N[t_n] == t)[0][0]
                    Phi[t, t_posn, k] = g[k, K[t_n, k]] - Phi[t_n, t_pos, K[t_n, k]] - C

    # Res
    for t in range(0, t_max):
        Res[t] = np.argmax(K[t])

    return K, Res


'''
@njit
def first_iteration(img, N, K, alpha):
    t_max = img.shape[0]
    for t in range(0, 2000): # t_max-1):

        t_n = [x for x in N[t] if x >= 0][0]
        foo = [0, 0]
        for k in [0, 1]:
            foo[k] = g(K[t_n], k,  alpha)

        K[t] = foo.index(max(foo))
        # print(f"{t} | t_n: {t_n},  foo: {foo},  K[{t}]: {K[t]}")
    return K


@njit
def next_iteration(Phi, img, N, K, alpha):

    K_next = K.copy()
    t_max = img.shape[0]
    for t in range(0, t_max):

        if t % 10000 == 0:
            print('[ ', t, ' | ', t_max, ']')

        foo = [0, 0]
        for k in [0, 1]:

            C, neighbours_amount = 0, 0
            for t_n in N[t]:
                if t_n != -1:
                    t_pos = np.where(N[t] == t_n)[0][0]
                    C += g(k, K[t_n], alpha) - Phi[t_pos, t, K[t_n]] + q(t, k, img)
                    neighbours_amount += 1
            C = C / neighbours_amount

            # Phi
            for t_n in N[t]:
                if t_n != -1:
                    t_pos = np.where(N[t] == t_n)[0][0]
                    t_pos2 = np.where(N[t_n] == t)[0][0]  # print('> :', t_n, ' | ', t, ' | ', N[t_n])

                    Phi[t, t_pos, k] = g(k, K[t_n], alpha) - Phi[t_n, t_pos2, K[t_n]] - C

        # K
        for t_n in N[t]:
            if t_n != -1:

                foo = [0, 0]
                t_pos = np.where(N[t] == t_n)[0][0]
                t_pos2 = np.where(N[t_n] == t)[0][0]

                for k in [0, 1]:
                    foo[k] = g(K[t_n], k, alpha) - Phi[t, t_pos, k] - Phi[t_n, t_pos2, K[t_n]]

                K_next[t_n] = foo.index(max(foo))
        # print(f"{t} | t_n: {t_n},  foo: {foo},  K[{t}]: {K[t]}")

    return K_next



'''








