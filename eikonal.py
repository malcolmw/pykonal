import heapq
import numpy as np

Nx, Ny, Nz = 20, 20, 20

def main():
    u = np.ones((Nx, Ny, Nz)) * float('inf')
    v = np.ones((Nx, Ny, Nz))
    for ix, iy, iz in [(ix, iy, iz) for ix in range(Nx)
                                    for iy in range(Ny)
                                    for iz in range(Nz)]:
        v[ix, iy, iz] = np.sqrt((ix-Nx/2)**2 + (iy-Ny/2)**2 + (iz-Nz)**2)
        v[ix, iy, iz] += np.random.rand(1) * 25
    plot(v)
    u = np.ma.masked_array(u, mask=False)
#####
    live = []
    heapq.heapify(live)
    start = [(0, 0, 0)]
    for s in start:
        u[s] = 0
        u.mask[s] = True
        heapq.heappush(live, (0, s))
#####
    while len(live) > 0:
        u, live = update(u, v, live)
    u = np.ma.getdata(u)
    plot(u)

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

def trace_ray(u, start, finish):
    h = 0.01
    grad = np.gradient(u)
    grad = np.stack((-1*grad[0], -1*grad[1]), axis=2)
    ray = [finish]
    xi, yi = finish + grad[finish] / np.linalg.norm(grad[finish])
    ray.append((xi, yi))
    for i in range(10000):
        dGdx = (grad[int(xi)+1, int(yi)][0]-grad[int(xi), int(yi)][0])
        dGdy = (grad[int(xi), int(yi)+1][1]-grad[int(xi), int(yi)][1])
        G0 = grad[int(xi), int(yi)]
        gradi = G0 + [xi % 1 * dGdx, yi % 1 * dGdy]
        dx, dy = gradi / np.linalg.norm(gradi) * min(h, 1/np.linalg.norm(gradi))
        xi += dx
        yi += dy
        ray.append((xi, yi))
    return(np.array(ray))

def plot_rays(u, rays):
    import matplotlib.pyplot as plt
    X, Y = np.meshgrid(range(Nx), range(Ny), indexing="ij")
    for ray in rays:
        plt.plot(ray[:,0], ray[:,1], "k")
        plt.scatter(X, Y, s=5, c=u, cmap=plt.get_cmap("jet"))
    plt.colorbar()
    plt.show()

def plot(u):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    pointsize = 5
    #grad = np.gradient(np.ma.getdata(u))
    #grad[0] *= -1
    #grad[1] *= -1
    #grad[3] *= -1
    X, Y, Z = np.meshgrid(range(Nx), range(Ny), range(Nz), indexing="ij")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.scatter(X, Y, Z, c=u, cmap=plt.get_cmap("jet"))
    #ax.quiver(X, Y, grad[0], grad[1], angles="xy", scale_units="xy", scale=10)
    #ax = fig.add_subplot(1, 3, 2)
    #ax.scatter(X, Y,
    #           c=np.stack((grad[0], grad[1]), axis=2)[...,0],
    #           s=pointsize,
    #           cmap=plt.get_cmap("jet"))
    #ax = fig.add_subplot(1, 3, 3)
    #cb = ax.scatter(X, Y,
    #                c=np.stack((grad[0], grad[1]), axis=2)[...,1],
    #                s=pointsize,
    #                cmap=plt.get_cmap("jet"))
    #fig.colorbar(cb)
    plt.show()

if __name__ == "__main__":
    main()
