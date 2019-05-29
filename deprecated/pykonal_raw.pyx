import heapq
import numpy as np

DTYPE = np.float32

def init_lists(vv):
    uu       = np.full(vv.shape, fill_value=np.inf, dtype=DTYPE)
    is_alive = np.full(vv.shape, fill_value=False, dtype=np.bool)
    is_far   = np.full(vv.shape, fill_value=True, dtype=np.bool)
    close    = []
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


def stencil(idx, idx_max):
    return (
            (idx[0] >= 0)
        and (idx[0] < idx_max[0])
        and (idx[1] >= 0)
        and (idx[1] < idx_max[1])
    )


def update(uu, vv, is_alive, close, is_far, dx, dy):
    '''
    The update algorithm to propagate the wavefront.

    uu - The travel-time field.
    vv - The velocity field.
    is_alive - Array of bool values indicating whether a node has a final value.
    close - A sorted heap of indices of nodes with temporary values.
    is_far - Array of bool values indicating whether a node has a temporary
             value.
    '''
    # Let Trial be the point in Close with the smallest value of u
    trial_idx = heapq.heappop(close)
    is_alive[trial_idx] = True
    i, j = trial_idx
    for nbr_idx in (idx for idx in ((i-1, j),
                                    (i+1, j),
                                    (i, j-1),
                                    (i, j+1))
                        if stencil(idx, is_alive.shape)
                        and not is_alive[idx]):
        nbr_i, nbr_j = nbr_idx
        u0 = uu[nbr_idx]
        x, y = nbr_i * dx, nbr_j * dy
        # Tag as Close all neighbours of Trial that are not Alive
        if nbr_idx not in close:
            heapq.heappush(close, nbr_idx)
        # If the neighbour is in Far, remove it from that list and add it to
        # Close
        is_far[nbr_idx] = False
        # Recompute the values of u at all Close neighbours of Trial by solving
        # the piecewise quadratic equation.
        ux = min(uu[nbr_i-1, nbr_j] if nbr_i-1 >= 0 else np.inf,
                 uu[nbr_i+1, nbr_j] if nbr_i+1 < vv.shape[0] else np.inf)
        uy = min(uu[nbr_i, nbr_j-1] if nbr_j-1 >= 0 else np.inf,
                 uu[nbr_i, nbr_j+1] if nbr_j+1 < vv.shape[1] else np.inf)

        ux, ddx2 = (ux, (1/dx)**2) if ux < u0 else (0, 0)
        uy, ddy2 = (uy, (1/dy)**2) if uy < u0 else (0, 0)
        slo2 = 1/vv[nbr_idx]**2

        a = ddx2 + ddy2
        if a == 0:
            print(f'WARNING :: a == 0 {nbr_idx}')
            continue
        b = -2 * (ddx2 * ux + ddy2 * uy)
        c = (ddx2 * (ux ** 2) + ddy2 * (uy ** 2) - slo2)
        if b ** 2 < 4 * a * c:
            print(f'WARNING :: determinant is negative {nbr_idx}')
            continue
        new = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        if np.isnan(new):
            print(nbr_idx,  a, -b, b**2-4*a*c, 2*a)
        uu[nbr_idx] = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    close.sort(key=lambda idx: uu[idx])
    #return (uu, is_alive, close, is_far)


if __name__ == '__main__':
    print('Not an executable')
