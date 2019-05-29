import heapq
import numpy as np
cimport numpy as np
cimport libc.math

DTYPE = np.float32

cdef float MAX_FLOAT = np.finfo(DTYPE).max

def init_lists(vv):
    uu       = np.full(vv.shape, fill_value=MAX_FLOAT, dtype=DTYPE)
    is_alive = np.full(vv.shape, fill_value=False, dtype=np.bool)
    close    = []
    is_far   = np.full(vv.shape, fill_value=True, dtype=np.bool)
    heapq.heapify(close)
    return (uu, is_alive, close, is_far)


def init_source(uu, close, is_far):
    ix, iy = 0, 0
    uu[ix, iy] = 0
    is_far[ix, iy] = False
    close.append([ix, iy])
    return (uu, close, is_far)


def pykonal(vv, dx=1, dy=1):
    uu, is_alive, close, is_far = init_lists(vv)
    init_source(uu, close, is_far)
    update(uu, vv, is_alive, close, is_far, dx, dy)
    return (uu)


cdef bint stencil(
        Py_ssize_t ix, Py_ssize_t iy, Py_ssize_t max_ix, Py_ssize_t max_iy
):
    return (
            (ix >= 0)
        and (ix < max_ix)
        and (iy >= 0)
        and (iy < max_iy)
    )


def update(
        float[:,:] uu,
        float[:,:] vv,
        np.ndarray[np.npy_bool, ndim=2, cast=True] is_alive,
        list close,
        np.ndarray[np.npy_bool, ndim=2, cast=True] is_far,
        float dx,
        float dy
):
    '''
    The update algorithm to propagate the wavefront.

    uu - The travel-time field.
    vv - The velocity field.
    is_alive - Array of bool values indicating whether a node has a final value.
    close - A sorted heap of indices of nodes with temporary values.
    is_far - Array of bool values indicating whether a node has a temporary
             value.
    '''
    cdef Py_ssize_t       i, iax
    cdef Py_ssize_t[4][2] nbrs
    cdef Py_ssize_t[2]    max_idx, nbr, trial_idx
    cdef Py_ssize_t[2]    switch
    cdef int              bord, ford, drxn
    cdef float            a, b, c, bfd, ffd
    cdef float[2]         aa, bb, cc, dd, dd2

    max_idx  = [is_alive.shape[0], is_alive.shape[1]]
    dd       = [dx, dy]
    dd2      = [dx**2, dy**2]
    dx2, dy2 = dx**2, dy**2
    for iax in range(2):
        assert dd[iax] > 0

    while len(close) > 0:
        # Let Trial be the point in Close with the smallest value of u
        close.sort(key=lambda idx: uu[idx[0], idx[1]])
        trial_idx = heapq.heappop(close)
        is_alive[trial_idx[0], trial_idx[1]] = True

        nbrs[0][0] = trial_idx[0] - 1
        nbrs[0][1] = trial_idx[1]
        nbrs[1][0] = trial_idx[0] + 1
        nbrs[1][1] = trial_idx[1]
        nbrs[2][0] = trial_idx[0]
        nbrs[2][1] = trial_idx[1] - 1
        nbrs[3][0] = trial_idx[0]
        nbrs[3][1] = trial_idx[1] + 1
        for i in range(4):
            nbr_ix = nbrs[i][0]
            nbr_iy = nbrs[i][1]
            nbr    = nbrs[i]
            if not stencil(nbr[0], nbr[1], max_idx[0], max_idx[1]) \
                    or is_alive[nbr[0], nbr[1]]:
                continue
            # Recompute the values of u at all Close neighbours of Trial
            # by solving the piecewise quadratic equation.
            for iax in range(2):
                switch = [0, 0]
                switch[iax] = 1
                if nbr[iax] > 1 \
                        and not is_far[nbr[0]-switch[0]*2, nbr[1]-switch[1]*2] \
                        and not is_far[nbr[0]-switch[0], nbr[1]-switch[1]]:
                    bord = 2
                    bfd  = (
                        3 * uu[nbr[0],             nbr[1]] \
                      - 4 * uu[nbr[0]-switch[0],   nbr[1]-switch[1]] \
                      +     uu[nbr[0]-switch[0]*2, nbr[1]-switch[1]*2]
                    ) / (2 * dd[iax])
                elif nbr[iax] > 0 \
                        and not is_far[nbr[0]-switch[0], nbr[1]-switch[1]]:
                    bord = 1
                    bfd  = (
                        uu[nbr[0],           nbr[1]]
                      - uu[nbr[0]-switch[0], nbr[1]-switch[1]]
                    ) / dd[iax]
                else:
                    bfd, bord = 0, 0
                if nbr[iax] < max_idx[iax] - 2 \
                        and not is_far[nbr[0]+switch[0]*2, nbr[1]+switch[1]*2] \
                        and not is_far[nbr[0]+switch[0], nbr[1]+switch[1]]:
                    ford = 2
                    ffd  = (
                        3 * uu[nbr[0],             nbr[1]] \
                      - 4 * uu[nbr[0]+switch[0],   nbr[1]+switch[1]] \
                      +     uu[nbr[0]+switch[0]*2, nbr[1]+switch[1]*2]
                    ) / (2 * dd[iax])
                elif nbr[iax] < max_idx[iax]-1 \
                        and not is_far[nbr[0]+switch[0], nbr[1]+switch[1]]:
                    ford = 1
                    ffd  = (
                        uu[nbr[0]+switch[0], nbr[1]+switch[1]]
                      - uu[nbr[0],           nbr[1]]
                    ) / dd[iax]
                else:
                    ffd, ford = 0, 0
                if bfd > -ffd:
                    order, drxn = bord, -1
                else:
                    order, drxn = ford, 1
                if order == 2:
                    aa[iax] = 9 / (4 * dd2[iax])
                    bb[iax] = (
                        6*uu[nbr[0]+2*drxn*switch[0], nbr[1]+2*drxn*switch[1]]
                     - 24*uu[nbr[0]+drxn*switch[0], nbr[1]+drxn*switch[1]]
                    ) / (4 * dd2[iax])
                    cc[iax] = (
                        uu[
                            nbr[0]+2*drxn*switch[0],
                            nbr[1]+2*drxn*switch[1]
                        ]**2 \
                        - 8 * uu[
                            nbr[0]+2*drxn*switch[0],
                            nbr[1]+2*drxn*switch[1]
                        ] * uu[
                            nbr[0]+drxn*switch[0],
                            nbr[1]+drxn*switch[1]
                        ]
                        + 16 * uu[
                            nbr[0]+drxn*switch[0],
                            nbr[1]+drxn*switch[1]
                        ]**2
                    ) / (4 * dd2[iax])
                elif order == 1:
                    aa[iax] = 1 / dd2[iax]
                    bb[iax] = -2 * uu[
                        nbr[0]+drxn*switch[0],
                        nbr[1]+drxn*switch[1]
                    ] / dd2[iax]
                    cc[iax] = uu[
                        nbr[0]+drxn*switch[0],
                        nbr[1]+drxn*switch[1]
                    ]**2 / dd2[iax]
                elif order == 0:
                    aa[iax], bb[iax], cc[iax] = 0, 0, 0
                else:
                    raise (Exception('Huh!?'))
            a = aa[0] + aa[1]
            b = bb[0] + bb[1]
            c = cc[0] + cc[1] - 1/vv[nbr[0], nbr[1]]**2
            if a == 0:
                print(f'WARNING(2) :: a == 0 {nbr[0]}, {nbr[1]}')
                continue
            if b ** 2 < 4 * a * c:
                # This may not be mathematically permissible
                uu[nbr[0], nbr[1]] = -b / (2 * a)
                print(
                    f'WARNING(2) :: determinant is negative {nbr[0]}, {nbr[1]}:'
                    f'{100*np.sqrt(4 * a * c - b**2)/(2*a)/uu[nbr[0], nbr[1]]}'
                )
            else:
                uu[nbr[0], nbr[1]] = (
                    -b + libc.math.sqrt(b ** 2 - 4 * a * c)
                ) / (2 * a)
            # Tag as Close all neighbours of Trial that are not Alive
            # If the neighbour is in Far, remove it from that list and add it to
            # Close
            if is_far[nbr[0], nbr[1]]:
                heapq.heappush(close, [nbr[0], nbr[1]])
                is_far[nbr[0], nbr[1]] = False

    return


if __name__ == '__main__':
    print('Not an executable')
