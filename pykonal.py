import heapq
import numpy as np
import seispy
import time

pi = np.pi

def init_lists(nodes):
    tt = np.full(nodes.shape[:-1], fill_value=np.inf)
    is_alive =  np.full(nodes.shape[:-1], fill_value=False)
    close = []
    heapq.heapify(close)
    is_far =  np.full(nodes.shape[:-1], fill_value=True)
    return(tt, is_alive, close, is_far)

def init_source(u, nodes, vm, is_alive, src, close, is_far, phase='p'):
    '''
    Initialize the travel-time grid given a source location.
    '''
    # Calculate the distance from source to each propagation grid node
    d = np.sqrt(np.sum(np.square(nodes.to_cartesian()-src.to_cartesian()), axis=3))
    # Determine the index of the node with minimum distance
    idx = np.unravel_index(np.argmin(d), d.shape)
    # Account for the case where multiple grid nodes are the same distance from the source
    for idx in np.argwhere(d == d[idx]):
        idx = tuple(idx)
        u[idx] = d[idx]/vm(phase, nodes[idx])
        is_far[idx] = False
        is_alive[idx] = True
        heapq.heappush(close, idx)
    close.sort()
    return(u, close, is_far)

def pykonal(vm, nodes, src):
    '''
    This is the main control loop.
    
    vm - VelocityModel object.
    nodes - Propagation grid nodes as GeographicCoordinates
    '''
    u, is_alive, close, is_far = init_lists(nodes)
    u, close, is_far = init_source(u, nodes, vm, is_alive, src, close, is_far)
    while len(close) > 0:
#     for i in range(50):
        try:
            u, is_alive, close, is_far = update(u, nodes, vm, is_alive, close, is_far)
        except:
            return (u, is_alive, close, is_far)
    return (u, is_alive, close, is_far)

def plot(u):
    xyz = pgrid.to_cartesian()
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', aspect=1)
    pts = ax.scatter(xyz[..., 0].flatten(), xyz[..., 1].flatten(), xyz[..., 2].flatten(), c=u.flatten())
    fig.colorbar(pts, ax=ax)
    plt.show()
    
def stencil(idx, idx_max):
    return (
            (idx[0] >= 0)
        and (idx[0] < idx_max[0])
        and (idx[1] >= 0)
        and (idx[1] < idx_max[1])
        and (idx[2] >= 0)
        and (idx[2] < idx_max[2])
    )

def get_backward_order(i, j, k, u, is_alive):
    if i - 2 >= 0 \
            and is_alive[i - 2,  j, k] \
            and is_alive[i - 1, j, k] \
            and u[i - 2, j, k] <= u[i - 1, j, k]:
        bord_r = 2
    elif i - 1 >= 0 \
            and is_alive[i - 1, j, k] \
            and u[i - 1, j, k] < u[i, j, k]:
        bord_r = 1
    else:
        bord_r = 0
    if j - 2 >= 0 \
            and is_alive[i, j - 2, k] \
            and is_alive[i, j - 1, k] \
            and u[i, j - 2, k] <= u[i, j - 1, k]:
        bord_t = 2
    elif j - 1 >= 0 \
            and is_alive[i, j - 1, k] \
            and u[i, j - 1, k] < u[i, j, k]:
        bord_t = 1
    else:
        bord_t = 0
    if k - 2 >= 0 \
            and is_alive[i, j, k - 2] \
            and is_alive[i, j, k - 1] \
            and u[i, j, k - 2] <= u[i, j, k - 1]:
        bord_p = 2
    elif k - 1 >= 0 \
            and is_alive[i , j, k - 1] \
            and u[i, j, k - 1] < u[i, j, k]:
        bord_p = 1
    else:
        bord_p = 0
#     bord_r, bord_t, bord_p = min(bord_r, 1), min(bord_t, 1), min(bord_p, 1)
    return(bord_r, bord_t, bord_p)

def get_forward_order(i, j, k, u, is_alive):
    if i + 2 < is_alive.shape[0] \
            and is_alive[i + 2, j, k] \
            and is_alive[i + 1, j, k] \
            and u[i + 2, j, k] <= u[i + 1, j, k]:
        ford_r = 2
    elif i + 1 < is_alive.shape[0] \
            and is_alive[i + 1, j, k] \
            and u[i + 1, j, k] < u[i, j, k]:
        ford_r = 1
    else:
        ford_r = 0
    if j + 2 < is_alive.shape[1] \
            and is_alive[i, j + 2, k] \
            and is_alive[i, j + 1, k] \
            and u[i, j + 2, k] <= u[i, j + 1, k]:
        ford_t = 2
    elif j + 1 < is_alive.shape[1] \
            and is_alive[i, j + 1, k] \
            and u[i, j + 1, k] < u[i, j, k]:
        ford_t = 1
    else:
        ford_t = 0
    if k + 2 < is_alive.shape[2] \
            and is_alive[i, j, k + 2] \
            and is_alive[i, j, k + 1] \
            and u[i, j, k + 2] <= u[i, j, k + 1]:
        ford_p = 2
    elif k + 1 < is_alive.shape[2] \
            and is_alive[i, j, k + 1] \
            and u[i, j, k + 1] < u[i, j, k]:
        ford_p = 1
    else:
        ford_p = 0
#     ford_r, ford_t, ford_p = min(ford_r, 1), min(ford_t, 1), min(ford_p, 1)
    return(ford_r, ford_t, ford_p)

def get_coeff(i, j, k, u, bord, ford, idx):
    order = (bord, ford)
#     print(f'getting {(bord, ford)}-update coefficients for node {(i, j, k)} along axis {idx}')
    idx_v = np.array([1 if i == idx else 0 for i in range(3)])
    idx_b1 = tuple((i, j, k) - idx_v)
    idx_b2 = tuple((i, j, k) - 2 * idx_v)
    idx_f1 = tuple((i, j, k) + idx_v)
    idx_f2 = tuple((i, j, k) + 2 * idx_v)
    
    if order == (2, 2):
        if u[idx_b2] - 4 * u[idx_b1] > u[idx_f2] - 4 * u[idx_f1]:
            c1 = 9
            c2 = 6 * u[idx_b2] - 24 * u[idx_b1]
            c3 = 16 * u[idx_b1] ** 2 \
                    - 8 * u[idx_b1] * u[idx_b2] \
                    + u[idx_b2] ** 2
            norm_ord = 2
        elif u[idx_b2] - 4 * u[idx_b1] < u[idx_f2] - 4 * u[idx_f1]:
            c1 = 9
            c2 = 6 * u[idx_f2] - 24 * u[idx_f1]
            c3 = 16 * u[idx_f1] ** 2 \
                    - 8 * u[idx_f1] * u[idx_f2] \
                    + u[idx_f2] ** 2
            norm_ord = 2
        else:
            raise (NotImplementedError('u[idx_b2] - 4 * u[idx_b1] == u[idx_f2] - 4 * u[idx_f1]'))
    elif order == (2, 1):
        if u[i, j, k] > u[idx_b2] - 4 * u[idx_b1] + 2 * u[idx_f1]:
            c1 = 9
            c2 = 6 * u[idx_b2] - 24 * u[idx_b1]
            c3 = 16 * u[idx_b1] ** 2 \
                    - 8 * u[idx_b1] * u[idx_b2] \
                    + u[idx_b2] ** 2
            norm_ord = 2
        elif u[i, j, k] < u[idx_b2] - 4 * u[idx_b1] + 2 * u[idx_f1]:
            c1 = 1
            c2 = -2 * u[idx_f1]
            c3 = u[idx_f1] ** 2
            norm_ord = 1
        else:
            raise (NotImplementedError('u[idx] == u[idx_b2] - 4 * u[idx_b1] + 2 * u[idx_f1]'))
    elif order == (2, 0):
        c1 = 9
        c2 = 6 * u[idx_b2] - 24 * u[idx_b1]
        c3 = 16 * u[idx_b1] ** 2 \
                - 8 * u[idx_b1] * u[idx_b2] \
                + u[idx_b2] ** 2
        norm_ord = 2
    elif order == (1, 2):
        if u[i, j, k] > u[idx_f2] - 4 * u[idx_f1] + 2 * u[idx_b1]:
            c1 = 9
            c2 = 6 * u[idx_f2] - 24 * u[idx_f1]
            c3 = 16 * u[idx_f1] ** 2 \
                    - 8 * u[idx_f1] * u[idx_f2] \
                    + u[idx_f2] ** 2
            norm_ord = 2
        elif u[i, j, k] < u[idx_f2] - 4 * u[idx_f1] + 2 * u[idx_b1]:
            c1 = 1
            c2 = -2 * u[idx_b1]
            c3 = u[idx_b1] ** 2
            norm_ord = 1
        else:
            raise (NotImplementedError('u[idx] == u[idx_f2] - 4 * u[idx_f1] + 2 * u[idx_b1]'))
    elif order == (1, 1):
        if u[idx_f1] > u[idx_b1]:
            c1 = 1
            c2 = -2 * u[idx_b1]
            c3 = u[idx_b1] ** 2
            norm_ord = 1
        elif u[idx_f1] < u[idx_b1]:
            c1 = 1
            c2 = 2 * u[idx_f1]
            c3 = u[idx_f1] ** 2
            norm_ord = 1
        else:
            raise (NotImplementedError('u[idx_f1] == u[idx_b1]'))
    elif order == (1, 0):
        c1 = 1
        c2 = -2 * u[idx_b1]
        c3 = u[idx_b1] ** 2
        norm_ord = 1
    elif order == (0, 2):
        c1 = 9
        c2 = 6 * u[idx_f2] - 24 * u[idx_f1]
        c3 = 16 * u[idx_f1] ** 2 \
                - 8 * u[idx_f1] * u[idx_f2] \
                + u[idx_f2] ** 2
        norm_ord = 2
    elif order == (0, 1):
        c1 = 1
        c2 = -2 * u[idx_f1]
        c3 = u[idx_f1] ** 2
        norm_ord = 1
    elif order == (0, 0):
        c1 = c2 = c3 = 0
        norm_ord = 0
    else:
        raise (NotImplementedError(f'order == {order}'))
#     print(f'update coeffecients are {(c1, c2, c3)}')
    c1 = c1 if not np.isinf(c1) else 0
    c2 = c2 if not np.isinf(c2) else 0
    c3 = c3 if not np.isinf(c3) else 0
    return(c1, c2, c3, norm_ord)

def update(u, nodes, vm, is_alive, close, is_far, phase='p'):
    '''
    The update algorithm to propagate the wavefront.
    
    u - The travel-time field.
    nodes - The propagation grid nodes as GeographicCoordinates.
    is_alive - Array of bool values indicating whether a node has a final value.
    close - A sorted heap of indices of nodes with temporary values.
    is_far - Array of bool values indicating whether a node has a temporary value.
    '''
    # Let Trial be the point in Close with the smallest value of u
    trial_idx = heapq.heappop(close)
    is_alive[trial_idx] = True
#     print(f'update {trial_idx}')
    i, j, k = trial_idx
    idx_max = is_alive.shape
    for nbr_idx in (idx for idx in ((i-1, j, k),
                                    (i+1, j, k),
                                    (i, j-1, k),
                                    (i, j+1, k),
                                    (i, j, k-1),
                                    (i, j, k+1))
                        if stencil(idx, idx_max)
                        and not is_alive[idx]):
        nbr_i, nbr_j, nbr_k = nbr_idx
        u0 = u[nbr_idx]
        lat, lon, depth = nbr_rx_geo = nodes[nbr_idx]
        rho, theta, phi = nbr_rx_geo.to_spherical()
        # Tag as Close all neighbours of Trial that are not Alive
        if nbr_idx not in close:
            heapq.heappush(close, nbr_idx)
        # If the neighbour is in Far, remove it from that list and add it to Close
        is_far[nbr_idx] = False
        # Recompute the values of u at all Close neighbours of Trial by solving the piecewise quadratic
        # equation.
        bord_r, bord_t, bord_p = get_backward_order(nbr_i, nbr_j, nbr_k, u, is_alive)
        ford_r, ford_t, ford_p =  get_forward_order(nbr_i, nbr_j, nbr_k, u, is_alive)

        a1, b1, c1, norm_ord = get_coeff(nbr_i, nbr_j, nbr_k, u, bord_r, ford_r, 0)
        norm = 1 / (2 * nodes.dr) ** 2 if norm_ord == 2 \
                else 1 / nodes.dr ** 2 if norm_ord == 1 \
                else 0
        a1, b1, c1 = (0, 0, 0) if norm == 0 \
                else (a1 * norm, b1 * norm, c1 * norm)

        a2, b2, c2, norm_ord = get_coeff(nbr_i, nbr_j, nbr_k, u, bord_t, ford_t, 1)
        norm = 1 / (2 * rho * nodes.dt) ** 2 if norm_ord == 2 \
                else 1 / (rho * nodes.dt) ** 2 if norm_ord == 1 \
                else 0
        a2, b2, c2 = (0, 0, 0) if norm == 0 \
                else (a2 * norm, b2 * norm, c2 * norm)

        a3, b3, c3, norm_ord = get_coeff(nbr_i, nbr_j, nbr_k, u, bord_p, ford_p, 2)
        norm = 1 / (2 * rho * np.sin(theta) * nodes.dp) ** 2 if norm_ord == 2 \
                else 1 / (rho * np.sin(theta) * nodes.dp) ** 2 if norm_ord == 1 \
                else 0
        a3, b3, c3 = (0, 0, 0) if norm == 0 \
                else (a3 * norm, b3 * norm, c3 * norm)
        
        slo = 1/vm(phase, nbr_rx_geo)

        aa = a1 + a2 + a3
        if aa == 0:
            print(f'WARNING :: aa == 0 {nbr_idx}')
            continue
        bb = b1 + b2 + b3
        cc = c1 + c2 + c3 - slo ** 2
        if bb ** 2 < 4 * aa * cc:
            print(f'WARNING :: determinant is negative {nbr_idx}: {bb**2 - 4 * aa * cc}')
            print(f'{a1, b1, c1}')
            print(f'{bord_r, bord_t, bord_p}')
            print(f'{ford_r, ford_t, ford_p}')
            raise(ValueError)
            continue
        new = (-bb + np.sqrt(bb ** 2 - 4 * aa * cc)) / (2 * aa)

        if np.isnan(new):
            print(nbr_idx,  aa, -bb, bb**2-4*aa*cc, 2*aa)
        if new < u[nbr_idx]:
            u[nbr_idx] = new
    close.sort(key=lambda idx: u[idx])
    return (u, is_alive, close, is_far)
