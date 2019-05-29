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


def pykonal_fo(vv, dx=1, dy=1):
    uu, is_alive, close, is_far = init_lists(vv)
    init_source(uu, close, is_far)
    update_fo(uu, vv, is_alive, close, is_far, dx, dy)
    return (uu)

def pykonal_mo(vv, dx=1, dy=1):
    uu, is_alive, close, is_far = init_lists(vv)
    init_source(uu, close, is_far)
    update_mo(uu, vv, is_alive, close, is_far, dx, dy)
    return (uu)

def pykonal_mo_x_only(vv, dx=1, dy=1):
    uu, is_alive, close, is_far = init_lists(vv)
    init_source(uu, close, is_far)
    update_mo_x_only(uu, vv, is_alive, close, is_far, dx, dy)
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


def update_fo(
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

    while len(close) > 0:
        # Let Trial be the point in Close with the smallest value of u
        close.sort(key=lambda idx: uu[idx[0], idx[1]])
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
            # Recompute the values of u at all Close neighbours of Trial
            # by solving the piecewise quadratic equation.
            if nbr_ix > 0:
                ux1 = uu[nbr_ix-1, nbr_iy]
            else:
                ux1 = MAX_FLOAT
            if nbr_ix < uu.shape[0] - 1:
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
            if nbr_iy < uu.shape[1] - 1:
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
            if a == 0:
                print(f'WARNING :: a == 0 {nbr_ix}, {nbr_iy}')
                continue
            b = -2 * (ddx2 * ux + ddy2 * uy)
            c = (ddx2 * (ux ** 2) + ddy2 * (uy ** 2) - slo2)
            if b ** 2 < 4 * a * c:
                unew = -b / (2 * a) # This may not be mathematically permissible...
                print(f'WARNING :: determinant is negative {nbr_ix}, {nbr_iy}: {100*np.sqrt(4 * a * c - b**2)/(2*a)/unew}')
            else:
                unew = (-b + libc.math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            uu[nbr_ix, nbr_iy] = unew
            # Tag as Close all neighbours of Trial that are not Alive
            # If the neighbour is in Far, remove it from that list and add it to
            # Close
            if is_far[nbr_ix, nbr_iy]:
                heapq.heappush(close, [nbr_ix, nbr_iy])
                is_far[nbr_ix, nbr_iy] = False

    return


def update_mo_x_only(
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
    dx2, dy2 = dx**2, dy**2

    while len(close) > 0:
        # Let Trial be the point in Close with the smallest value of u
        close.sort(key=lambda idx: uu[idx[0], idx[1]])
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
            # Recompute the values of u at all Close neighbours of Trial
            # by solving the piecewise quadratic equation.
            if nbr_ix > 1 \
                    and not is_far[nbr_ix-2, nbr_iy] \
                    and not is_far[nbr_ix-1, nbr_iy]:
                bo_x  =   2
                bfd_x = (
                    3 * uu[nbr_ix, nbr_iy] \
                  - 4 * uu[nbr_ix-1, nbr_iy] \
                  +     uu[nbr_ix-2, nbr_iy]
                ) / (2 * dx)
            elif nbr_ix > 0 and not is_far[nbr_ix-1, nbr_iy]:
                bo_x = 1
                bfd_x = (uu[nbr_ix, nbr_iy] - uu[nbr_ix-1, nbr_iy]) / dx
            else:
                bo_x, bfd_x = 0, 0
            if nbr_ix < max_ix-2 \
                    and not is_far[nbr_ix+2, nbr_iy] \
                    and not is_far[nbr_ix+1, nbr_iy]:
                fo_x  =   2
                ffd_x = (
                    3 * uu[nbr_ix, nbr_iy] \
                  - 4 * uu[nbr_ix+1, nbr_iy] \
                  +     uu[nbr_ix+2, nbr_iy]
                ) / (2 * dx)
            elif nbr_ix < max_ix-1 and not is_far[nbr_ix+1, nbr_iy]:
                fo_x = 1
                ffd_x = (uu[nbr_ix+1, nbr_iy] - uu[nbr_ix, nbr_iy]) / dx
            else:
                fo_x, ffd_x = 0, 0
            if bfd_x > -ffd_x:
                o_x = -bo_x
            else:
                o_x = fo_x
            if o_x == -2:
                a_x = 9 / (4 * dx2)
                b_x = (6 * uu[nbr_ix-2, nbr_iy] - 24 * uu[nbr_ix-1, nbr_iy]) \
                        / (4 * dx2)
                c_x = (
                         uu[nbr_ix-2, nbr_iy]**2 \
                  -  8 * uu[nbr_ix-2, nbr_iy] * uu[nbr_ix-1, nbr_iy]
                  + 16 * uu[nbr_ix-1, nbr_iy]**2
                ) / (4 * dx2)
            elif o_x == -1:
                a_x = 1 / dx2
                b_x = -2 * uu[nbr_ix-1, nbr_iy] / dx2
                c_x = uu[nbr_ix-1, nbr_iy]**2 / dx2
            elif o_x == 0:
                a_x, b_x, c_x = 0, 0, 0
            elif o_x == 1:
                a_x = 1 / dx2
                b_x = -2 * uu[nbr_ix+1, nbr_iy] / dx2
                c_x = uu[nbr_ix+1, nbr_iy]**2 / dx2
            elif o_x == 2:
                a_x = 9 / (4 * dx2)
                b_x = (6 * uu[nbr_ix+2, nbr_iy] - 24 * uu[nbr_ix+1, nbr_iy]) \
                        / (4 * dx2)
                c_x = (
                         uu[nbr_ix+2, nbr_iy]**2 \
                  -  8 * uu[nbr_ix+2, nbr_iy] * uu[nbr_ix+1, nbr_iy]
                  + 16 * uu[nbr_ix+1, nbr_iy]**2
                ) / (4 * dx2)
            else:
                raise (Exception('What!?'))

            if nbr_iy > 0 and not is_far[nbr_ix, nbr_iy-1]:
                bo_y = 1
                bfd_y = (uu[nbr_ix, nbr_iy] - uu[nbr_ix, nbr_iy-1]) / dy
            else:
                bo_y, bfd_y = 0, 0
            if nbr_iy < max_iy-1 and not is_far[nbr_ix, nbr_iy+1]:
                fo_y = 1
                ffd_y = (uu[nbr_ix, nbr_iy+1] - uu[nbr_ix, nbr_iy]) / dy
            else:
                fo_y, ffd_y = 0, 0
            if bfd_y > -ffd_y:
                o_y = -bo_y
            else:
                o_y = fo_y
            if o_y == -1:
                a_y = 1 / dy2
                b_y = -2 * uu[nbr_ix, nbr_iy-1] / dy2
                c_y = uu[nbr_ix, nbr_iy-1]**2 / dy2
            elif o_y == 0:
                a_y, b_y, c_y = 0, 0, 0
            else:
                a_y = 1 / dy2
                b_y = -2 * uu[nbr_ix, nbr_iy+1] / dy2
                c_y = uu[nbr_ix, nbr_iy+1]**2 / dy2

            a = a_x + a_y
            b = b_x + b_y
            c = c_x + c_y - 1/vv[nbr_ix, nbr_iy]**2
            #print(f'({nbr_ix}, {nbr_iy})')
            #print(f'{o_x}, {a_x}, {b_x}, {c_x}')
            #print(f'{o_y}, {a_y}, {b_y}, {c_y}')
            #print(f'{a}, {b}, {c}')
            #print('-----------------------------------')

            if a == 0:
                print(f'WARNING :: a == 0 {nbr_ix}, {nbr_iy}')
                continue
            if b ** 2 < 4 * a * c:
                unew = -b / (2 * a) # This may not be mathematically permissible
                print(f'WARNING :: determinant is negative {nbr_ix}, {nbr_iy}: '
                      f'{100*np.sqrt(4 * a * c - b**2)/(2*a)/unew}')
            else:
                unew = (-b + libc.math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            uu[nbr_ix, nbr_iy] = unew
            # Tag as Close all neighbours of Trial that are not Alive
            # If the neighbour is in Far, remove it from that list and add it to
            # Close
            if is_far[nbr_ix, nbr_iy]:
                heapq.heappush(close, [nbr_ix, nbr_iy])
                is_far[nbr_ix, nbr_iy] = False

    return


def update_mo(
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
    cdef float[2] aa, bb, cc, dd, dd2
    cdef Py_ssize_t i
    cdef Py_ssize_t[4][2] nbrs
    cdef Py_ssize_t[2]    nbr, max_idx
    cdef Py_ssize_t[3]    switch

    max_ix, max_iy = is_alive.shape[0], is_alive.shape[1]
    max_idx = [max_ix, max_iy]
    dd = [dx, dy]
    dd2 = [dx**2, dy**2]
    dx2, dy2 = dx**2, dy**2

    while len(close) > 0:
        # Let Trial be the point in Close with the smallest value of u
        close.sort(key=lambda idx: uu[idx[0], idx[1]])
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
            nbr    = nbrs[i]
            if not stencil(nbr_ix, nbr_iy, max_ix, max_iy) \
                    or is_alive[nbr_ix, nbr_iy]:
                continue
            u0 = uu[nbr_ix, nbr_iy]
            # Recompute the values of u at all Close neighbours of Trial
            # by solving the piecewise quadratic equation.
            for iax in range(2):
                switch = [0, 0, 0]
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

            #if nbr_ix > 1 \
            #        and not is_far[nbr_ix-2, nbr_iy] \
            #        and not is_far[nbr_ix-1, nbr_iy]:
            #    bo_x  =   2
            #    bfd_x =   (
            #        3 * uu[nbr_ix, nbr_iy] \
            #      - 4 * uu[nbr_ix-1, nbr_iy] \
            #      +     uu[nbr_ix-2, nbr_iy]
            #    ) / (2 * dx)
            #elif nbr_ix > 0 and not is_far[nbr_ix-1, nbr_iy]:
            #    bo_x = 1
            #    bfd_x = (uu[nbr_ix, nbr_iy] - uu[nbr_ix-1, nbr_iy]) / dx
            #else:
            #    bo_x, bfd_x = 0, 0
            #if nbr_ix < max_ix-2 \
            #        and not is_far[nbr_ix+2, nbr_iy] \
            #        and not is_far[nbr_ix+1, nbr_iy]:
            #    fo_x  =   2
            #    ffd_x = (
            #        3 * uu[nbr_ix, nbr_iy] \
            #      - 4 * uu[nbr_ix+1, nbr_iy] \
            #      +     uu[nbr_ix+2, nbr_iy]
            #    ) / (2 * dx)
            #elif nbr_ix < max_ix-1 and not is_far[nbr_ix+1, nbr_iy]:
            #    fo_x = 1
            #    ffd_x = (uu[nbr_ix+1, nbr_iy] - uu[nbr_ix, nbr_iy]) / dx
            #else:
            #    fo_x, ffd_x = 0, 0
            #if bfd_x > -ffd_x:
            #    o_x = -bo_x
            #else:
            #    o_x = fo_x
            #if o_x == -2:
            #    a_x = 9 / (4 * dx2)
            #    b_x = (6 * uu[nbr_ix-2, nbr_iy] - 24 * uu[nbr_ix-1, nbr_iy]) \
            #            / (4 * dx2)
            #    c_x = (
            #             uu[nbr_ix-2, nbr_iy]**2 \
            #      -  8 * uu[nbr_ix-2, nbr_iy] * uu[nbr_ix-1, nbr_iy]
            #      + 16 * uu[nbr_ix-1, nbr_iy]**2
            #    ) / (4 * dx2)
            #elif o_x == -1:
            #    a_x = 1 / dx2
            #    b_x = -2 * uu[nbr_ix-1, nbr_iy] / dx2
            #    c_x = uu[nbr_ix-1, nbr_iy]**2 / dx2
            #elif o_x == 0:
            #    a_x, b_x, c_x = 0, 0, 0
            #elif o_x == 1:
            #    a_x = 1 / dx2
            #    b_x = -2 * uu[nbr_ix+1, nbr_iy] / dx2
            #    c_x = uu[nbr_ix+1, nbr_iy]**2 / dx2
            #elif o_x == 2:
            #    a_x = 9 / (4 * dx2)
            #    b_x = (6 * uu[nbr_ix+2, nbr_iy] - 24 * uu[nbr_ix+1, nbr_iy]) \
            #            / (4 * dx2)
            #    c_x = (
            #             uu[nbr_ix+2, nbr_iy]**2 \
            #      -  8 * uu[nbr_ix+2, nbr_iy] * uu[nbr_ix+1, nbr_iy]
            #      + 16 * uu[nbr_ix+1, nbr_iy]**2
            #    ) / (4 * dx2)
            #else:
            #    raise (Exception('What!?'))

            #if nbr_iy > 0 and not is_far[nbr_ix, nbr_iy-1]:
            #    bo_y = 1
            #    bfd_y = (uu[nbr_ix, nbr_iy] - uu[nbr_ix, nbr_iy-1]) / dy
            #else:
            #    bo_y, bfd_y = 0, 0
            #if nbr_iy < max_iy-1 and not is_far[nbr_ix, nbr_iy+1]:
            #    fo_y = 1
            #    ffd_y = (uu[nbr_ix, nbr_iy+1] - uu[nbr_ix, nbr_iy]) / dy
            #else:
            #    fo_y, ffd_y = 0, 0
            #if bfd_y > -ffd_y:
            #    o_y = -bo_y
            #else:
            #    o_y = fo_y
            #if o_y == -1:
            #    a_y = 1 / dy2
            #    b_y = -2 * uu[nbr_ix, nbr_iy-1] / dy2
            #    c_y = uu[nbr_ix, nbr_iy-1]**2 / dy2
            #elif o_y == 0:
            #    a_y, b_y, c_y = 0, 0, 0
            #else:
            #    a_y = 1 / dy2
            #    b_y = -2 * uu[nbr_ix, nbr_iy+1] / dy2
            #    c_y = uu[nbr_ix, nbr_iy+1]**2 / dy2

            #a = a_x + a_y
            #b = b_x + b_y
            #c = c_x + c_y - 1/vv[nbr_ix, nbr_iy]**2
            #if a == 0:
            #    print(f'WARNING(1) :: a == 0 {nbr_ix}, {nbr_iy}')
            #    continue
            #if b ** 2 < 4 * a * c:
            #    unew = -b / (2 * a) # This may not be mathematically permissible
            #    print(f'WARNING(1) :: determinant is negative {nbr_ix}, {nbr_iy}: '
            #          f'{100*np.sqrt(4 * a * c - b**2)/(2*a)/unew}')
            #else:
            #    unew = (-b + libc.math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            #uu[nbr_ix, nbr_iy] = unew
            a = aa[0] + aa[1]
            b = bb[0] + bb[1]
            c = cc[0] + cc[1] - 1/vv[nbr_ix, nbr_iy]**2
            if a == 0:
                print(f'WARNING(2) :: a == 0 {nbr_ix}, {nbr_iy}')
                continue
            if b ** 2 < 4 * a * c:
                unew = -b / (2 * a) # This may not be mathematically permissible
                print(f'WARNING(2) :: determinant is negative {nbr_ix}, {nbr_iy}: '
                      f'{100*np.sqrt(4 * a * c - b**2)/(2*a)/unew}')
            else:
                unew = (-b + libc.math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
            uu[nbr_ix, nbr_iy] = unew
            # Tag as Close all neighbours of Trial that are not Alive
            # If the neighbour is in Far, remove it from that list and add it to
            # Close
            if is_far[nbr_ix, nbr_iy]:
                heapq.heappush(close, [nbr_ix, nbr_iy])
                is_far[nbr_ix, nbr_iy] = False

    return


if __name__ == '__main__':
    print('Not an executable')
