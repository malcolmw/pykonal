# distutils: language = c++

import numpy as np
cimport numpy as np
cimport libc.math
from libcpp.vector cimport vector as cpp_vector


DTYPE = np.float32
cdef float MAX_FLOAT = np.finfo(DTYPE).max 
cdef struct Index2D:
    Py_ssize_t ix, iy
        

cdef void sift_up(cpp_vector[Index2D]& idxs, float[:,:] uu, Py_ssize_t j_start):
    '''Doc string'''
    cdef Py_ssize_t j, j_child, j_end, j_right
    cdef Index2D idx_child, idx_right, idx_new
    
    j_end = idxs.size()
    j = j_start
    idx_new = idxs[j_start]
    # Bubble up the smaller child until hitting a leaf.
    j_child = 2 * j_start + 1 # leftmost child position
    while j_child < j_end:
        # Set childpos to index of smaller child.
        j_right = j_child + 1
        idx_child, idx_right = idxs[j_child], idxs[j_right]
        if j_right < j_end and not uu[idx_child.ix, idx_child.iy] < uu[idx_right.ix, idx_right.iy]:
            j_child = j_right
        # Move the smaller child up.
        idxs[j] = idxs[j_child]
        j = j_child
        j_child = 2 * j + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    idxs[j] = idx_new
    sift_down(idxs, uu, j_start, j)


cdef void sift_down(cpp_vector[Index2D]& idxs, float[:,:] uu, Py_ssize_t j_start, Py_ssize_t j):
    '''Doc string'''
    cdef Py_ssize_t j_parent
    cdef Index2D idx_new, idx_parent
    
    idx_new = idxs[j]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while j > j_start:
        j_parent = (j - 1) >> 1
        idx_parent = idxs[j_parent]
        if uu[idx_new.ix, idx_new.iy] < uu[idx_parent.ix, idx_parent.iy]:
            idxs[j] = idx_parent
            j = j_parent
            continue
        break
    idxs[j] = idx_new


cdef void heap_push(cpp_vector[Index2D]& idxs, float[:,:] uu, Index2D idx):
    '''Push item onto heap, maintaining the heap invariant.'''
    idxs.push_back(idx)
    sift_down(idxs, uu, 0, idxs.size()-1)

cdef Index2D heap_pop(cpp_vector[Index2D]& idxs, float[:,:] uu):
    '''Pop the smallest item off the heap, maintaining the heap invariant.'''
    cdef Index2D last, idx_return
    
    last = idxs[idxs.size()-1]
    idxs.pop_back()
    if idxs.size() > 0:
        idx_return = idxs[0]
        idxs[0] = last
        sift_up(idxs, uu, 0)
        return (idx_return)
    return (last)


def init_lists(vv):
    cdef cpp_vector[Index2D] close
    uu       = np.full(vv.shape, fill_value=MAX_FLOAT, dtype=DTYPE)
    is_alive = np.full(vv.shape, fill_value=False, dtype=np.bool)
    is_far   = np.full(vv.shape, fill_value=True, dtype=np.bool)
    return (uu, is_alive, close, is_far)


cdef void init_source(
    float[:,:] uu, 
    cpp_vector[Index2D]& close, 
    np.ndarray[np.npy_bool, ndim=2, cast=True] is_far
):
    cdef Index2D idx
    idx.ix, idx.iy = 0, 0
    uu[idx.ix, idx.iy] = 0
    is_far[idx.ix, idx.iy] = False
    heap_push(close, uu, idx)


cdef bint stencil(
        Py_ssize_t ix, Py_ssize_t iy, Py_ssize_t max_ix, Py_ssize_t max_iy
):
    return (
            (ix >= 0)
        and (ix < max_ix)
        and (iy >= 0)
        and (iy < max_iy)
    )


cdef void update(
        float[:,:] uu,
        float[:,:] vv,
        np.ndarray[np.npy_bool, ndim=2, cast=True] is_alive,
        cpp_vector[Index2D] close,
        np.ndarray[np.npy_bool, ndim=2, cast=True] is_far,
        float dx,
        float dy
):
    '''The update algorithm to propagate the wavefront. '''
    cdef Index2D          idx, trial_idx
    cdef Py_ssize_t       i, iax
    cdef Py_ssize_t[4][2] nbrs
    cdef Py_ssize_t[2]    max_idx, nbr
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
        trial_idx = heap_pop(close, uu)
        trial_ix, trial_iy = trial_idx.ix, trial_idx.iy
        is_alive[trial_ix, trial_iy] = True

        nbrs[0][0] = trial_ix - 1
        nbrs[0][1] = trial_iy
        nbrs[1][0] = trial_ix + 1
        nbrs[1][1] = trial_iy
        nbrs[2][0] = trial_ix
        nbrs[2][1] = trial_iy - 1
        nbrs[3][0] = trial_ix
        nbrs[3][1] = trial_iy + 1
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
                        and not is_far[nbr[0]-switch[0], nbr[1]-switch[1]] \
                        and uu[nbr[0]-switch[0]*2, nbr[1]-switch[1]*2] <= uu[nbr[0]-switch[0], nbr[1]-switch[1]]:
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
                        and not is_far[nbr[0]+switch[0],   nbr[1]+switch[1]] \
                        and uu[nbr[0]+switch[0]*2, nbr[1]+switch[1]*2] <= uu[nbr[0]+switch[0], nbr[1]+switch[1]]:
                    ford = 2
                    ffd  = (
                      - 3 * uu[nbr[0],             nbr[1]] \
                      + 4 * uu[nbr[0]+switch[0],   nbr[1]+switch[1]] \
                      -     uu[nbr[0]+switch[0]*2, nbr[1]+switch[1]*2]
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
                     - 24*uu[nbr[0]+  drxn*switch[0], nbr[1]  +drxn*switch[1]]
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
#                 print(f'WARNING(2) :: a == 0 {nbr[0]}, {nbr[1]}')
                continue
            if b ** 2 < 4 * a * c:
                # This may not be mathematically permissible
                uu[nbr[0], nbr[1]] = -b / (2 * a)
#                 print(
#                     f'WARNING(2) :: determinant is negative {nbr[0]}, {nbr[1]}:'
#                     f'{100*np.sqrt(4 * a * c - b**2)/(2*a)/uu[nbr[0], nbr[1]]}'
#                 )
            else:
                uu[nbr[0], nbr[1]] = (
                    -b + libc.math.sqrt(b ** 2 - 4 * a * c)
                ) / (2 * a)
            # Tag as Close all neighbours of Trial that are not Alive
            # If the neighbour is in Far, remove it from that list and add it to
            # Close
            if is_far[nbr[0], nbr[1]]:
                idx.ix, idx.iy = nbr_ix, nbr_iy
                heap_push(close, uu, idx)
                is_far[nbr[0], nbr[1]] = False


def pykonal(vv):
    cdef cpp_vector[Index2D] close
    cdef np.ndarray[float, ndim=2] uu
    
    dx, dy = 1, 1
    uu, is_alive, close, is_far = init_lists(vv)
    init_source(uu, close, is_far)
    update(uu, vv, is_alive, close, is_far, dx, dy)
    return (uu)
