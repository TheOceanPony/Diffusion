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
     return np.zeros(img.shape[0])
    # return np.random.randint( 2, size=(img.shape[0],2), dtype=np.uint8 )


def init_g(alpha):
    return np.array([[alpha, 0], [0, alpha]])


def init_q(img):

    t_max = img.shape[0]
    c = np.array([[0., 255., 0.], [0., 0., 255.]])
    q = np.zeros((t_max, 2))

    for t in range(0, t_max):
        for k in [0,1]:
            if np.linalg.norm( img[t]-c[k] ) > np.linalg.norm( img[t]-c[abs(k - 1)] ):
                q[t, k] = -1
            else:
                q[t, k] = 0

    return q


# --------------------------------------------


@njit
def iteration(N, Phi, g, q, t_max):
    K = np.zeros((t_max,2), dtype=np.uint8)

    for k in [0, 1]:
        for t in range(0, t_max):
        
            # K*
            for t_n in N[t]:
                if t_n != -1:
                    foo = [-1, -1]
                    for k_n in [0, 1]:
                        t_pos = np.where(N[t_n] == t)[0][0]
                        t_posn = np.where(N[t] == t_n)[0][0]
                        foo[k_n] = g[k, k_n] - Phi[t_n, t_pos, k_n]

                    K[t_n, k] = foo.index(max(foo))

            # C
            C, nb_amnt = q[t, k], 0
            for t_n in N[t]:
                if t_n != -1:
                    t_pos = np.where(N[t_n] == t)[0][0]
                    t_posn = np.where(N[t] == t_n)[0][0]
                    C += g[k, K[t_n, k]] - Phi[t, t_posn, k] - Phi[t_n, t_pos, K[t_n, k]]
                    nb_amnt += 1
            C = C/nb_amnt

            # Phi
            for t_n in N[t]:
                if t_n != -1:
                    t_posn = np.where(N[t] == t_n)[0][0]
                    t_pos = np.where(N[t_n] == t)[0][0]
                    Phi[t, t_posn, k] = g[k, K[t_n, k]] - Phi[t_n, t_pos, K[t_n, k]] - C


    return Phi
        

def reconstruction(Phi, N, g, q, Res, t_max):
    for t in range(0, t_max):

        t_n = [x for x in N[t] if x >= 0][0] # Existing neighbour
        
        t_posn = np.where(N[t] == t_n)[0][0]
        t_pos = np.where(N[t_n] == t)[0][0] 

        foo1 = np.array([-1, -1])
        for k in [0, 1]:

            foo2 = np.array([-1, -1])
            
            for k_n in [0, 1]:

                foo2[k_n] = g[k, k_n] - Phi[t, t_posn, k] - Phi[t_n, t_pos, k_n]

            foo1[k] = np.max(foo2)

        Res[t] = np.argmax(foo1)

    
    # q check
    for k in [0, 1]:
        _sum = 0
        for t in range(0, t_max):
            _sum += q[t, k]

            temp = 0
            for t_n in N[t]:
                if t_n != -1:
                    t_posn = np.where(N[t] == t_n)[0][0] 
                    temp += Phi[t, t_posn, k]

            _sum += temp        

        print(f"{k, t} | sum: {_sum}")


    # G check
    for k in [0, 1]:
        for t in range(0, t_max):
        
            g_max = []
            for t_n in N[t]:
                if t_n != -1:

                    foo = []
                    t_posn = np.where(N[t] == t_n)[0][0]
                    t_pos = np.where(N[t_n] == t)[0][0] 
                    for k_n in [0, 1]:
                       foo.append( g[k, k_n] - Phi[t, t_posn, k] - Phi[t_n, t_pos, k_n] )

                    g_max.append( max(foo) )
            

            if t % 1000 == 0:
                print(f"[{t}, {k}]  |  g_tt': {g_max}")
            
                    
    return Res
