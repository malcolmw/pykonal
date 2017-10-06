import heapq
import numpy as np
import scipy.interpolate
import seispy

pi = np.pi

Nx, Ny, Nz = 20, 20, 20
VELOCITY_MODEL = "/Users/malcolcw/Projects/Shared/Velocity/FANG2016/original/"\
                 "VpVs.dat"

def main():
    vgrid = prep_vmod()
    u = np.ones(vgrid["velocity"].shape) * float('inf')
    u = np.ma.masked_array(u, mask=False)
#####
    live = []
    heapq.heapify(live)
    start = [(16, 31, 15)]
    for s in start:
        u[s] = 0
        u.mask[s] = True
        heapq.heappush(live, (0, s))
#####
    while len(live) > 0:
        u, live = update(u, vgrid, live)
    u = np.ma.getdata(u)
    plot_uv(u, vgrid)
    #rays = [trace_ray_runge_kutta(u, vgrid, start[0], (12, 9, 12))]
    #plot_rays(u, rays)

def prep_vmod():
    vm = seispy.velocity.VelocityModel(VELOCITY_MODEL,
                                       fmt="FANG")
    grid = vm.v_type_grids[1][1]["grid"]
    sc = seispy.coords.SphericalCoordinates(8)
    theta_max = grid.theta0 + (grid.ntheta - 1) * grid.dtheta
    phi_max = grid.phi0 + (grid.nphi - 1) * grid.dphi
    rho_max = grid.rho0 + (grid.nrho - 1) * grid.drho
    sc[...] = [(grid.rho0, theta_max, grid.phi0),
               (grid.rho0, theta_max, phi_max),
               (grid.rho0, grid.theta0, grid.phi0),
               (grid.rho0, grid.theta0, phi_max),
               (rho_max, theta_max, grid.phi0),
               (rho_max, theta_max, phi_max),
               (rho_max, grid.theta0, grid.phi0),
               (rho_max, grid.theta0, phi_max)]
    cc0 = sc.to_cartesian()

    cc0 = cc0.rotate(grid.phi0, 0, 0)
    cc0 = cc0.rotate(0, theta_max, 0)
    cc0 = cc0.rotate(pi/2, 0, 0)

    nx, ny, nz = grid.nphi, grid.ntheta, grid.nrho

    cc = seispy.coords.as_cartesian(
            [(x, y, z) for x in np.linspace(min(cc0[:,0]), max(cc0[:,0]), nx)
                       for y in np.linspace(min(cc0[:,1]), max(cc0[:,1]), ny)
                       for z in np.linspace(min(cc0[:,2]), max(cc0[:,2]), nz)])
    cc = cc.rotate(-pi/2, 0, 0)
    cc = cc.rotate(0, -theta_max, 0)
    cc = cc.rotate(-grid.phi0, 0, 0)

    v = np.array([vm(1, 1, lat, lon, depth)
                    for (lat, lon, depth) in cc.to_geographic()])

    cc = cc.rotate(grid.phi0, 0, 0)
    cc = cc.rotate(0, theta_max, 0)
    cc = cc.rotate(pi/2, 0, 0)

    cc = np.reshape(cc, (nx, ny, nz, 3))
    v = np.reshape(v, (nx, ny, nz))

    vgrid = {"velocity": v,
             "coords": cc,
             "dx": (np.max(cc[...,0]) - np.min(cc[...,0])) / (nx - 1),
             "dy": (np.max(cc[...,1]) - np.min(cc[...,1])) / (ny - 1),
             "dz": (np.max(cc[...,2]) - np.min(cc[...,2])) / (nz - 1)}


#####
#    import matplotlib.pyplot as plt
#    from mpl_toolkits.mplot3d import Axes3D
#    fig = plt.figure()
#    ax = fig.add_subplot(1, 1, 1, projection="3d")
#    ax.scatter(cc[..., 0], cc[...,1], cc[...,2],
#               s=1,
#               c=v,
#               cmap=plt.get_cmap("jet_r"))
#    ax.set_xlabel("E")
#    ax.set_ylabel("N")
#    ax.set_zlabel("R")
#    ax.set_zlim((6271, 6521))
#    plt.show()
######
    return(vgrid)

def update(u, vgrid, live):
    v = vgrid["velocity"]
    dx, dy, dz = vgrid["dx"], vgrid["dy"], vgrid["dz"]
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
        ux = min(u0[max(i-1, 0), j, k],
                 u0[min(i+1, u0.shape[0]-1), j, k])
        uy = min(u0[i, max(j-1, 0), k],
                 u0[i, min(j+1, u0.shape[1]-1), k])
        uz = min(u0[i, j, max(k-1, 0)],
                 u0[i, j, min(k+1, u0.shape[2]-1)])
        A = 1/dx**2 + 1/dy**2 + 1/dz**2
        B = -2*(ux/dx**2 + uy/dy**2 + uz/dz**2)
        C = (ux/dx)**2 + (uy/dy)**2 + (uz/dz)**2 - 1/v[i, j, k]**2
        D = B**2 - 4 * A * C
        if D >= 0:
            u[i, j, k] = (-B + np.sqrt(D)) / (2 * A)
        else:
            A1 = 1/dx**2 + 1/dy**2
            B1 = -2*(ux/dx**2 + uy/dy**2)
            C1 = (ux/dx)**2 + (uy/dy)**2 - 1/v[i, j, k]**2
            D1 = B1**2 - 4 * A1 * C1

            A2 = 1/dx**2 + 1/dz**2
            B2 = -2*(ux/dx**2 + uz/dz**2)
            C2 = (ux/dx)**2 + (uz/dz)**2 - 1/v[i, j, k]**2
            D2 = B2**2 - 4 * A2 * C2

            A3 = 1/dy**2 + 1/dz**2
            B3 = -2*(uy/dy**2 + uz/dz**2)
            C3 = (uy/dy)**2 + (uz/dz)**2 - 1/v[i, j, k]**2
            D3 = B3**2 - 4 * A3 * C3

            u1 = (-B1 + np.sqrt(D1)) / (2 * A1)  if D1 >= 0\
                    else ux + dx / v[i, j, k] if ux < uy\
                    else uy + dy / v[i, j, k]

            u2 = (-B2 + np.sqrt(D2)) / (2 * A2)  if D2 >= 0\
                    else ux + dx / v[i, j, k] if ux < uz\
                    else uz + dz / v[i, j, k]

            u3 = (-B3 + np.sqrt(D3)) / (2 * A3)  if D3 >= 0\
                    else uy + dy / v[i, j, k] if uy < uz\
                    else uz + dz / v[i, j, k]

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

def trace_ray_runge_kutta(u, vgrid, start, finish):
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
        print(np.sqrt(np.sum(np.square(ray[-2]-start)))-
                np.sqrt(np.sum(np.square(ray[-1]-start))),
                np.sqrt(np.sum(np.square(ray[-1]-start))))
        if np.sqrt(np.sum(np.square(ray[-1]-start))) > \
                np.sqrt(np.sum(np.square(ray[-2]-start))):
            break
        print(ray[-1])
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
        print(1/np.linalg.norm(gradi))
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
    #cb = ax.scatter(X, Y, Z,
    #                c=u,
    #                cmap=plt.get_cmap("jet_r"),
    #                alpha=0.2)
    #fig.colorbar(cb)
    plt.show()

def plot_uv(u, vgrid):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    pointsize = 5
    grad = np.gradient(np.ma.getdata(u))
    grad[0] *= -1
    grad[1] *= -1
    grad[2] *= -1
    cc = vgrid["coords"]
    fig = plt.figure()
    #ax = fig.add_subplot(1, 2, 1, projection="3d")
    #ax.set_title("Velocity model")
    #cb = ax.scatter(cc[...,0], cc[...,1], cc[...,2],
    #                c=vgrid["velocity"],
    #                cmap=plt.get_cmap("jet_r"))
    #ax.set_zlim((6251, 6521))
    #ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_title("Travel-time field")
    ax.set_zlim((6251, 6521))
    cb = ax.scatter(cc[...,0], cc[...,1], cc[...,2], c=u, cmap=plt.get_cmap("jet_r"))
    plt.show()

if __name__ == "__main__":
    main()
