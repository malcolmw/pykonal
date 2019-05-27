import heapq
import numpy as np
cimport numpy as np
cimport libc.math

DTYPE = np.float32

cdef float MAX_FLOAT = np.finfo(DTYPE).max


cdef void heap_sort(Py_ssize_t[:,:] idxs, float[:,:] uu):
    '''
    This is adapted from Numerical Recipes in C
    '''
    cdef Py_ssize_t i, j, k, l, n
    cdef Py_ssize_t ix1, ix2, iy1, iy2
    cdef Py_ssize_t tmp_ix, tmp_iy
    n = len(idxs) - 1
    if n < 2:
        return
    k = (n >> 1) + 1
    l = n
    while True:
        if k > 1:
            k -= 1
            tmp_ix, tmp_iy = idxs[k][0], idxs[k][1]
        else:
            tmp_ix, tmp_iy = idxs[l][0], idxs[l][1]
            idxs[l] = idxs[1]
            l -= 1
            if l == 1:
                idxs[1][0], idxs[1][1] = tmp_ix, tmp_iy
                break
        i = k
        j = k + k
        while j <= l:
            ix1, iy1 = idxs[j][0], idxs[j][1]
            ix2, iy2 = idxs[j+1][0], idxs[j+1][1]
            if j < l and uu[ix1][iy1] < uu[ix2][iy2]:
                j += 1
            ix1, iy1 = idxs[j]
            if uu[tmp_ix][tmp_iy] < uu[ix1][iy1]:
                idxs[i] = idxs[j]
                i = j
                j <<= 1
            else:
                break
        idxs[i][0], idxs[i][1] = tmp_ix, tmp_iy

def init_lists(vv):
    MAX_FLOAT = np.finfo(DTYPE).max
    uu       = np.full(vv.shape, fill_value=MAX_FLOAT, dtype=DTYPE)
    is_alive = np.full(vv.shape, fill_value=False, dtype=np.bool)
    close    = []
    is_far   = np.full(vv.shape, fill_value=True, dtype=np.bool)
    heapq.heapify(close)
    return (uu, is_alive, close, is_far)


def init_source(uu, close, is_far):
    idx = (0, 0)
    uu[idx] = 0
    is_far[idx] = False
    #close.append(idx)
    heapq.heappush(close, idx)
    close.sort()
    return (uu, close, is_far)


def pykonal(vv):
    dx, dy = 1, 1
    uu, is_alive, close, is_far = init_lists(vv)
    init_source(uu, close, is_far)
    while len(close) > 0:
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
    cdef Py_ssize_t ix, iy
    cdef Py_ssize_t trial_ix, trial_iy
    cdef Py_ssize_t nbr_ix, nbr_iy
    cdef Py_ssize_t max_ix, max_iy
    cdef float u0
    cdef float ux, ux1, ux2, uy, uy1, uy2
    cdef float ddx2, ddy2
    cdef float slo2, a, b, c
    cdef float unew
    cdef Py_ssize_t i
    cdef Py_ssize_t[4][2] nbrs

    max_ix, max_iy = is_alive.shape[0], is_alive.shape[1]

    # Let Trial be the point in Close with the smallest value of u
    trial_idx = heapq.heappop(close)
    trial_ix, trial_iy = trial_idx[0], trial_idx[1]
    is_alive[trial_ix, trial_iy] = True

    nbrs[0][0] = trial_ix-1
    nbrs[0][1] = trial_iy
    nbrs[1][0] = trial_ix+1
    nbrs[1][1] = trial_iy
    nbrs[2][0] = trial_ix
    nbrs[2][1] = trial_iy-1
    nbrs[3][0] = trial_ix
    nbrs[3][1] = trial_iy+1
    for i in range(4):
        nbr_ix = nbrs[i][0]
        nbr_iy = nbrs[i][1]
        if not stencil(nbr_ix, nbr_iy, max_ix, max_iy) \
                or is_alive[nbr_ix, nbr_iy]:
            continue
        u0 = uu[nbr_ix, nbr_iy]
        # Tag as Close all neighbours of Trial that are not Alive
        # If the neighbour is in Far, remove it from that list and add it to
        # Close
        if is_far[nbr_ix, nbr_iy]:
            heapq.heappush(close, (nbr_ix, nbr_iy))
            is_far[nbr_ix, nbr_iy] = False

        # Recompute the values of u at all Close neighbours of Trial by solving
        # the piecewise quadratic equation.
        if nbr_ix > 0:
            ux1 = uu[nbr_ix-1, nbr_iy]
        else:
            ux1 = MAX_FLOAT
        if nbr_ix < vv.shape[0] - 1:
            ux2 = uu[nbr_ix+1, nbr_iy]
        else:
            ux2 = MAX_FLOAT
        if ux1 < ux2:
            ux = ux1
        else:
            ux = ux2

        if nbr_iy > 0:
            uy1 = uu[nbr_ix, nbr_iy-1]
        else:
            uy1 = MAX_FLOAT
        if nbr_iy < vv.shape[1] - 1:
            uy2 = uu[nbr_ix, nbr_iy+1]
        else:
            uy2 = MAX_FLOAT
        if uy1 < uy2:
            uy = uy1
        else:
            uy = uy2

        if ux < u0:
            ddx2 = (1/dx)**2
        else:
            ux, ddx2 = 0, 0
        if uy < u0:
            ddy2 = (1/dy)**2
        else:
            uy, ddy2 = 0, 0

        slo2 = 1/vv[nbr_ix, nbr_iy]**2

        a = ddx2 + ddy2
#        if a == 0:
#            print(f'WARNING :: a == 0 {nbr_ix}, {nbr_iy}')
#            continue
        b = -2 * (ddx2 * ux + ddy2 * uy)
        c = (ddx2 * (ux ** 2) + ddy2 * (uy ** 2) - slo2)
#        if b ** 2 < 4 * a * c:
#            print(f'WARNING :: determinant is negative {nbr_ix}, {nbr_iy}')
#            continue
        unew = (-b + libc.math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
#        if np.isnan(unew):
#            print(nbr_ix, nbr_iy,  a, -b, b**2-4*a*c, 2*a)
        uu[nbr_ix, nbr_iy] = unew
    close.sort(key=lambda idx: uu[idx[0], idx[1]])


if __name__ == '__main__':
    print('Not an executable')
