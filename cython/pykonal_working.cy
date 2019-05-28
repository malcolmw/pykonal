import heapq
import numpy as np
cimport numpy as np

DTYPE = np.float32

def init_lists(vv):
    uu       = np.full(vv.shape, fill_value=np.inf, dtype=DTYPE)
    is_alive = np.full(vv.shape, fill_value=False, dtype=np.bool)
    close    = []
    is_far   = np.full(vv.shape, fill_value=True, dtype=np.bool)
    heapq.heapify(close)
    return (uu, is_alive, close, is_far)


def init_source(uu, close, is_far):
    idx = (0, 0)
    uu[idx] = 0
    is_far[idx] = False
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
        is_alive,
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
    cdef float ux, uy
    cdef float ddx2, ddy2
    cdef float slo2, a, b, c
    cdef float unew
    max_ix, max_iy = is_alive.shape[0], is_alive.shape[1]

    # Let Trial be the point in Close with the smallest value of u
    trial_idx = heapq.heappop(close)
    trial_ix, trial_iy = trial_idx[0], trial_idx[1]
    is_alive[trial_ix, trial_iy] = True

    for (nbr_ix, nbr_iy) in ((ix, iy) for (ix, iy)
                             in  ((trial_ix-1, trial_iy),
                                  (trial_ix+1, trial_iy),
                                  (trial_ix,   trial_iy-1),
                                  (trial_ix,   trial_iy+1))
                             if stencil(ix, iy, max_ix, max_iy)
                             and not is_alive[ix, iy]):
        nbr_idx = (nbr_ix, nbr_iy)
        #nbr_ix, nbr_iy = nbr_idx[0], nbr_idx[1]
        u0 = uu[nbr_ix, nbr_iy]
        # Tag as Close all neighbours of Trial that are not Alive
        if nbr_idx not in close:
            heapq.heappush(close, nbr_idx)
        # If the neighbour is in Far, remove it from that list and add it to
        # Close
        #is_far[nbr_idx] = False
        is_far[nbr_ix, nbr_iy] = False
        # Recompute the values of u at all Close neighbours of Trial by solving
        # the piecewise quadratic equation.
        ux = min(uu[nbr_ix-1, nbr_iy] if nbr_ix-1 >= 0 else np.inf,
                 uu[nbr_ix+1, nbr_iy] if nbr_ix+1 < vv.shape[0] else np.inf)
        uy = min(uu[nbr_ix, nbr_iy-1] if nbr_iy-1 >= 0 else np.inf,
                 uu[nbr_ix, nbr_iy+1] if nbr_iy+1 < vv.shape[1] else np.inf)

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
        if a == 0:
            print(f'WARNING :: a == 0 {nbr_idx}')
            continue
        b = -2 * (ddx2 * ux + ddy2 * uy)
        c = (ddx2 * (ux ** 2) + ddy2 * (uy ** 2) - slo2)
        if b ** 2 < 4 * a * c:
            print(f'WARNING :: determinant is negative {nbr_idx}')
            continue
        unew = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        if np.isnan(unew):
            print(nbr_idx,  a, -b, b**2-4*a*c, 2*a)
        uu[nbr_ix, nbr_iy] = unew
    close.sort(key=lambda idx: uu[idx[0], idx[1]])


if __name__ == '__main__':
    print('Not an executable')
