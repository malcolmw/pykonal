import heapq
import numpy as np
import scipy.interpolate

Nx, Ny, Nz = 20, 20, 20

def main():
    u = np.ones((Nx, Ny, Nz)) * float('inf')
    v = np.ones((Nx, Ny, Nz))
    for ix, iy, iz in [(ix, iy, iz) for ix in range(Nx)
                                    for iy in range(Ny)
                                    for iz in range(Nz)]:
        #v[ix, iy, iz] = np.sqrt((ix-Nx/2)**2 + (iy-Ny/2)**2 + (iz-Nz)**2)
        v[ix, iy, iz] = Nz-iz
        #v[ix, iy, iz] += np.random.rand(1) * 10
    #plot(v)
    u = np.ma.masked_array(u, mask=False)
#####
    live = []
    heapq.heapify(live)
    start = [(9, 9, 7)]
    for s in start:
        u[s] = 0
        u.mask[s] = True
        heapq.heappush(live, (0, s))
#####
    while len(live) > 0:
        u, live = update(u, v, live)
    u = np.ma.getdata(u)
    plot_uv(u, v)
    rays = [trace_ray_runge_kutta(u, start[0], (0, 0, 19))]
    rays += [trace_ray_runge_kutta(u, start[0], (0, 9, 19))]
    rays += [trace_ray_runge_kutta(u, start[0], (9, 9, 19))]
    rays += [trace_ray_runge_kutta(u, start[0], (9, 0, 19))]
    rays += [trace_ray_runge_kutta(u, start[0], (19, 0, 19))]
    rays += [trace_ray_runge_kutta(u, start[0], (0, 19, 19))]
    rays += [trace_ray_runge_kutta(u, start[0], (19, 19, 19))]
    rays += [trace_ray_runge_kutta(u, start[0], (14, 14, 19))]
    plot_rays(u, rays)

def update(u, v, live):
    h = 1
    u0 = np.ma.getdata(u)
    _, active = heapq.heappop(live)
    near = [(i, j, k) for (i, j, k) in [(active[0]-1, active[1], active[2]),
                                        (active[0]+1, active[1], active[2]),
                                        (active[0], active[1]-1, active[2]),
                                        (active[0], active[1]+1, active[2]),
                                        (active[0], active[1], active[2]-1),
                                        (active[0], active[1], active[2]+1)]
                   if 0 <= i < u0.shape[0]
                   and 0 <= j < u0.shape[1]
                   and 0 <= k < u0.shape[2]
                   and not u.mask[i, j, k]]
    for (i, j, k) in near:
        #if not (0 <= i < u0.shape[0] and 0 <= j < u0.shape[1]) or u.mask[i, j]:
        #    continue
        hv = h/v[i, j, k]
        ux = min(u0[max(i-1, 0), j, k],
                 u0[min(i+1, u0.shape[0]-1), j, k])
        uy = min(u0[i, max(j-1, 0), k],
                 u0[i, min(j+1, u0.shape[1]-1), k])
        uz = min(u0[i, j, max(k-1, 0)],
                 u0[i, j, min(k+1, u0.shape[2]-1)])
        #isinf = (isxinf, isyinf, iszinf) = (np.isinf(ux), np.isinf(uy), np.isinf(uz))
        d = (ux + uy + uz)**2 - 3 * (ux**2 + uy**2 + uz**2 - hv**2)
        if d >= 0:
            u[i, j, k] = 1/3 * (ux + uy + uz + np.sqrt(d))
        else:
            d1 = (ux + uy)**2 - 2*(ux**2 + uy**2 - hv**2)
            d2 = (ux + uz)**2 - 2*(ux**2 + uz**2 - hv**2)
            d3 = (uy + uz)**2 - 2*(uy**2 + uz**2 - hv**2)
            u1 = 1/2 * (ux + uy + np.sqrt(d1)) if d1 >= 0 else min(ux, uy) + hv
            u2 = 1/2 * (ux + uz + np.sqrt(d2)) if d2 >= 0 else min(ux, uz) + hv
            u3 = 1/2 * (uy + uz + np.sqrt(d3)) if d3 >= 0 else min(uy, uz) + hv
            u[i, j, k] = min(u1, u2, u3)
    u.mask[active] = True
    indices = [l[1] for l in live]
    for ijk in near:
        if ijk in indices:
            index = indices.index(ijk)
            live[index] = (u[ijk], ijk)
        else:
            heapq.heappush(live, (u[ijk], ijk))
    live.sort()
    return(u, live)

def interp_grad(grad, p):
    ix0 = min(max(int(p[0]), 0), Nx-1)
    ix1 = min(max(ix0+1, 0), Nx-1)
    dx = p[0]-ix0
    iy0 = min(max(int(p[1]), 0), Ny-1)
    iy1 = min(max(iy0+1, 0), Ny-1)
    dy = p[1]-iy0
    iz0 = min(max(int(p[2]), 0), Nz-1)
    iz1 = min(max(iz0+1, 0), Nz-1)
    dz = p[2]-iz0

    Gx000 = grad[ix0, iy0, iz0, 0]
    Gx001 = grad[ix0, iy0, iz1, 0]
    Gx010 = grad[ix0, iy1, iz0, 0]
    Gx011 = grad[ix0, iy1, iz1, 0]
    Gx100 = grad[ix1, iy0, iz0, 0]
    Gx101 = grad[ix1, iy0, iz1, 0]
    Gx110 = grad[ix1, iy1, iz0, 0]
    Gx111 = grad[ix1, iy1, iz1, 0]
    Gx00 = Gx000 + (Gx100-Gx000) * dx
    Gx01 = Gx001 + (Gx101-Gx001) * dx
    Gx10 = Gx010 + (Gx110-Gx010) * dx
    Gx11 = Gx011 + (Gx111-Gx011) * dx
    Gx0 = Gx00 + (Gx10 - Gx00) * dy
    Gx1 = Gx01 + (Gx11 - Gx01) * dy
    Gx = Gx0 + (Gx1 - Gx0) * dz

    Gy000 = grad[ix0, iy0, iz0, 1]
    Gy001 = grad[ix0, iy0, iz1, 1]
    Gy010 = grad[ix0, iy1, iz0, 1]
    Gy011 = grad[ix0, iy1, iz1, 1]
    Gy100 = grad[ix1, iy0, iz0, 1]
    Gy101 = grad[ix1, iy0, iz1, 1]
    Gy110 = grad[ix1, iy1, iz0, 1]
    Gy111 = grad[ix1, iy1, iz1, 1]
    Gy00 = Gy000 + (Gy100-Gy000) * dx
    Gy01 = Gy001 + (Gy101-Gy001) * dx
    Gy10 = Gy010 + (Gy110-Gy010) * dx
    Gy11 = Gy011 + (Gy111-Gy011) * dx
    Gy0 = Gy00 + (Gy10 - Gy00) * dy
    Gy1 = Gy01 + (Gy11 - Gy01) * dy
    Gy = Gy0 + (Gy1 - Gy0) * dz

    Gz000 = grad[ix0, iy0, iz0, 2]
    Gz001 = grad[ix0, iy0, iz1, 2]
    Gz010 = grad[ix0, iy1, iz0, 2]
    Gz011 = grad[ix0, iy1, iz1, 2]
    Gz100 = grad[ix1, iy0, iz0, 2]
    Gz101 = grad[ix1, iy0, iz1, 2]
    Gz110 = grad[ix1, iy1, iz0, 2]
    Gz111 = grad[ix1, iy1, iz1, 2]
    Gz00 = Gz000 + (Gz100-Gz000) * dx
    Gz01 = Gz001 + (Gz101-Gz001) * dx
    Gz10 = Gz010 + (Gz110-Gz010) * dx
    Gz11 = Gz011 + (Gz111-Gz011) * dx
    Gz0 = Gz00 + (Gz10 - Gz00) * dy
    Gz1 = Gz01 + (Gz11 - Gz01) * dy
    Gz = Gz0 + (Gz1 - Gz0) * dz

    return(np.array((Gx, Gy, Gz)))

def trace_ray_runge_kutta(u, start, finish):
    h = 0.1
    grad0 = np.stack(np.gradient(u), axis=3)
    grad = lambda p: interp_grad(grad0, p)
    ray = np.array([finish])
    while np.sqrt(np.sum(np.square(ray[-1]-start))) > h:
        g0 = grad(ray[-1])
        g0 /= np.linalg.norm(g0)
        p1 = ray[-1] - h/2 * g0
        g1 = grad(p1)
        g1 /= np.linalg.norm(g1)
        p2 = ray[-1] - h/2 * g1
        g2 = grad(p2)
        g2 /= np.linalg.norm(g2)
        p3 = ray[-1] - h * g2
        g3 = grad(p3)
        g3 /= np.linalg.norm(g3)
        ray = np.vstack((ray,
                         ray[-1] - (h/6 * g0\
                                  + h/3 * g1\
                                  + h/3 * g2\
                                  + h/6 * g3)))
        #print(np.sqrt(np.sum(np.square(ray[-2]-start)))-
        #        np.sqrt(np.sum(np.square(ray[-1]-start))),
        #        np.sqrt(np.sum(np.square(ray[-1]-start))))
        if np.sqrt(np.sum(np.square(ray[-1]-start))) > \
                np.sqrt(np.sum(np.square(ray[-2]-start))):
            break
        #print(ray[-1])
    return(ray)

def trace_ray_euler(u, start, finish):
    """
    """
    h = 0.01
    grad = np.gradient(u)
    grad = np.stack((-1*grad[0], -1*grad[1], -1*grad[2]), axis=3)
    ray = [finish]
    xi, yi, zi = finish + grad[finish] / np.linalg.norm(grad[finish])
    ray.append((xi, yi, zi))
    while np.sqrt((xi-start[0])**2+(yi-start[1])**2+(zi-start[2])**2) > 0.1:
        dGdx = (grad[int(xi)+1, int(yi), int(zi)][0]\
                - grad[int(xi), int(yi), int(zi)][0])
        dGdy = (grad[int(xi), int(yi)+1, int(zi)][1]\
                - grad[int(xi), int(yi), int(zi)][1])
        dGdz = (grad[int(xi), int(yi), int(zi)+1][2]\
                - grad[int(xi), int(yi), int(zi)][2])
        G0 = grad[int(xi), int(yi), int(zi)]
        gradi = G0 + [xi % 1 * dGdx, yi % 1 * dGdy, zi % 1 * dGdz]
        dx, dy, dz = gradi / np.linalg.norm(gradi) \
                   * min(h, 1/np.linalg.norm(gradi))
        #print(1/np.linalg.norm(gradi))
        xi += dx
        yi += dy
        zi += dz
        if np.sqrt((xi-start[0])**2+(yi-start[1])**2+(zi-start[2])**2)\
                > np.sqrt((ray[-1][0]-start[0])**2\
                        + (ray[-1][1]-start[1])**2\
                        + (ray[-1][2]-start[2])**2):
            return(np.array(ray))
        ray.append((xi, yi, zi))
    return(np.array(ray))

def plot_rays(u, rays):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    X, Y, Z = np.meshgrid(range(Nx),
                          range(Ny),
                          range(Nz),
                          indexing="ij")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for ray in rays:
        ax.plot(ray[:,0], ray[:,1], ray[:,2])
    cb = ax.scatter(X, Y, Z,
                    c=u,
                    cmap=plt.get_cmap("jet_r"),
                    alpha=0.2)
    fig.colorbar(cb)
    plt.show()

def plot_uv(u, v):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    pointsize = 5
    grad = np.gradient(np.ma.getdata(u))
    grad[0] *= -1
    grad[1] *= -1
    grad[2] *= -1
    X, Y, Z = np.meshgrid(range(Nx), range(Ny), range(Nz), indexing="ij")
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.set_title("Velocity model")
    cb = ax.scatter(X, Y, Z, c=v, cmap=plt.get_cmap("jet_r"))
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.set_title("Travel-time field")
    cb = ax.scatter(X, Y, Z, c=u, cmap=plt.get_cmap("jet_r"))
    #ax = fig.add_subplot(1, 3, 3, projection="3d")
    #ax.scatter(X, Y, Z,
    #           c=np.stack((grad[0], grad[1], grad[2]), axis=3)[...,0],
    #           s=pointsize,
    #           cmap=plt.get_cmap("jet"))
    #ax = fig.add_subplot(1, 3, 3)
    #cb = ax.scatter(X, Y,
    #                c=np.stack((grad[0], grad[1]), axis=2)[...,1],
    #                s=pointsize,
    #                cmap=plt.get_cmap("jet"))
    #fig.colorbar(cb)
    plt.show()

def plot(u):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    pointsize = 5
    grad = np.gradient(np.ma.getdata(u))
    grad[0] *= -1
    grad[1] *= -1
    grad[2] *= -1
    X, Y, Z = np.meshgrid(range(Nx), range(Ny), range(Nz), indexing="ij")
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    cb = ax.scatter(X, Y, Z, c=u, cmap=plt.get_cmap("jet"))
    #ax.quiver(X, Y, Z, grad[0], grad[1], grad[2], angles="xy", scale_units="xy", scale=10)
    #ax.quiver(X, Y, Z, grad[0], grad[1], grad[2], length=0.5, normalize=True)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(X, Y, Z,
               c=np.stack((grad[0], grad[1], grad[2]), axis=3)[...,0],
               s=pointsize,
               cmap=plt.get_cmap("jet"))
    #ax = fig.add_subplot(1, 3, 3)
    #cb = ax.scatter(X, Y,
    #                c=np.stack((grad[0], grad[1]), axis=2)[...,1],
    #                s=pointsize,
    #                cmap=plt.get_cmap("jet"))
    #fig.colorbar(cb)
    plt.show()

if __name__ == "__main__":
    main()
