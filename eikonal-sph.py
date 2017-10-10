import heapq
import numpy as np
import seispy
import time

pi = np.pi

VELOCITY_MODEL = "data/VpVs.dat"
SOURCE = (33.0, -116.0, 12.0)

def main():
    vgrid = prep_vmod()
    u = np.ones(vgrid["velocity"].shape) * float('inf')
    u = np.ma.masked_array(u, mask=False)
#####
    live = []
    heapq.heapify(live)
    source = (0, 15, 15)
    u[source] = 0
    u.mask[source] = True
    heapq.heappush(live, (u[source], source))
#####
    t = time.time()
    print("Starting update")
    while len(live) > 0:
        u, live = update(u, vgrid, live)
    print("Update took %.3f s" % (time.time() - t))
    u = np.ma.getdata(u)
    grad = nabla(u, vgrid)
    print(grad.shape)
    vgrid["velocity"] = grad[...,0]
    plot_vgrid(vgrid)
    vgrid["velocity"] = grad[...,1]
    plot_vgrid(vgrid)
    vgrid["velocity"] = grad[...,2]
    plot_vgrid(vgrid)

def nabla(u, vgrid):
    grid = vgrid["grid"]
    sc = vgrid["coords"]
    dudr = np.concatenate(([(u[1] - u[0])/grid.drho],
                           (u[2:] - 2*u[1:-1] + u[:-2])/grid.drho**2,
                           [(u[-1] - u[-2])/grid.drho]),
                           axis=0)
    dudt = np.concatenate((np.reshape([(u[:,1] - u[:,0])/(sc[:,0,:,0]*grid.dtheta)], (sc.shape[0], 1, sc.shape[2])),
                            (u[:,2:] - 2*u[:,1:-1] + u[:,:-2])\
                                    /(sc[:,1:-1,:,0]*grid.dtheta)**2,
                           np.reshape([(u[:,-1] - u[:,-2])/(sc[:,-1,:,0]*grid.dtheta)], (sc.shape[0], 1, sc.shape[2]))),
                           axis=1)

    dudp = np.concatenate((np.reshape([(u[:,:,1] - u[:,:,0])\
                                /(sc[:,:,0,0]*np.cos(sc[:,:,0,1])*grid.dphi)],(sc.shape[0], sc.shape[1], 1)),
                           (u[:,:,2:] - 2*u[:,:,1:-1] + u[:,:,:-2])\
                                /(sc[:,:,1:-1,0]*np.cos(sc[:,:,1:-1,1])*grid.dphi)**2,
                           np.reshape([(u[:,:,-2] - u[:,:,-1])\
                                /(sc[:,:,-1,0]*np.cos(sc[:,:,-1,1])*grid.dphi)], (sc.shape[0], sc.shape[1], 1))),
                         axis=2)
    return(np.stack((dudr, dudt, dudp), axis=3))

def update(u, vgrid, live):
    v = vgrid["velocity"]
    grid = vgrid["grid"]
    drho, dtheta, dphi = grid.drho, grid.dtheta, grid.dphi
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
        rho, theta, phi = vgrid["coords"][i, j, k]
        ur = min(u0[max(i-1, 0), j, k],
                 u0[min(i+1, u0.shape[0]-1), j, k])
        ut = min(u0[i, max(j-1, 0), k],
                 u0[i, min(j+1, u0.shape[1]-1), k])
        up = min(u0[i, j, max(k-1, 0)],
                 u0[i, j, min(k+1, u0.shape[2]-1)])
        ur, ddr2 = (ur, 1/drho**2) if ur < u[i, j, k] else (0, 0)
        ut, ddt2 = (ut, 1/(rho*dtheta)**2) if ut < u[i, j, k] else (0, 0)
        up, ddp2 = (up, 1/(rho*np.cos(theta)*dphi)**2) if up < u[i, j, k] else (0, 0)

        A = ddr2 + ddt2 + ddp2
        B = -2 * (ur*ddr2 + ut*ddt2 + up*ddp2)
        C = (ur**2)*ddr2 + (ut**2)*ddt2 + (up**2)*ddp2 - 1/v[i, j, k]**2

        #if B**2 < 4*A*C:
        #    print("A, B, C = %g, %g, %g" % (A, B, C), ddr2, ddt2, ddp2)
        #    print(ur, ut, up, u[i, j, k])
        #    print(active, (i, j, k))
        u[i, j, k] = (-B + max(0, np.sqrt(B**2 - 4*A*C))) / (2*A)
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

def prep_vmod():
    vm = seispy.velocity.VelocityModel(VELOCITY_MODEL,
                                       fmt="FANG")
    grid = vm.v_type_grids[1][1]["grid"]
    sc = seispy.coords.SphericalCoordinates(8)
    sc = seispy.coords.as_spherical([(rho, theta, phi) 
            for rho in np.linspace(grid.rho0,
                                   grid.rho0 + (grid.nrho-1)*grid.drho,
                                   grid.nrho)
            for theta in np.linspace(grid.theta0,
                                     grid.theta0 + (grid.ntheta-1)*grid.dtheta,
                                     grid.ntheta)
            for phi in np.linspace(grid.phi0,
                                   grid.phi0 + (grid.nphi-1)*grid.dphi,
                                   grid.nphi)])
    sc = np.reshape(sc, (grid.nrho,
                         grid.ntheta,
                         grid.nphi, 3)).astype(np.float64)
    grid.drho = np.float64(grid.drho)
    grid.dtheta = np.float64(grid.dtheta)
    grid.dphi = np.float64(grid.dphi)
    vgrid = {"velocity": vm.v_type_grids[1][1]["data"],
             "coords": sc,
             "grid": grid}
    return(vgrid)

def plot_vgrid(vgrid):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    cc = vgrid["coords"].to_cartesian().rotate(vgrid["grid"].phi0,
                                               vgrid["grid"].theta0,
                                               pi/2)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    cax = ax.scatter(cc[..., 0], cc[...,1], cc[...,2],
                     s=1,
                     c=vgrid["velocity"],
                     #vmin=-1.5,
                     #vmax=1.5,
                     cmap=plt.get_cmap("jet_r"))
    fig.colorbar(cax)
    plt.show()

def interp_grad(grad, vgrid, xyz):
    dx, dy, dz = vgrid["dx"], vgrid["dy"], vgrid["dz"]
    ix = xyz[0] / dx
    iy = xyz[1] / dy
    iz = (vgrid["zmax"] - xyz[2]) / dz
    ix0, ix1 = max(int(ix), 0), min(int(ix) + 1, vgrid["nx"] - 1)
    iy0, iy1 = max(int(iy), 0), min(int(iy) + 1, vgrid["ny"] - 1)
    iz0, iz1 = max(int(iz), 0), min(int(iz) + 1, vgrid["nz"] - 1)

    Gx000, Gy000, Gz000 = grad[ix0, iy0, iz0]
    Gx001, Gy001, Gz001 = grad[ix0, iy0, iz1]
    Gx010, Gy010, Gz010 = grad[ix0, iy1, iz0]
    Gx011, Gy011, Gz011 = grad[ix0, iy1, iz1]
    Gx100, Gy100, Gz100 = grad[ix1, iy0, iz0]
    Gx101, Gy101, Gz101 = grad[ix1, iy0, iz1]
    Gx110, Gy110, Gz110 = grad[ix1, iy1, iz0]
    Gx111, Gy111, Gz111 = grad[ix1, iy1, iz1]

    Gx00 = Gx000 + (Gx100-Gx000) * (ix - ix0)
    Gx01 = Gx001 + (Gx101-Gx001) * (ix - ix0)
    Gx10 = Gx010 + (Gx110-Gx010) * (ix - ix0)
    Gx11 = Gx011 + (Gx111-Gx011) * (ix - ix0)
    Gx0 = Gx00 + (Gx10 - Gx00) * (iy - iy0)
    Gx1 = Gx01 + (Gx11 - Gx01) * (iy - iy0)
    Gx = Gx0 + (Gx1 - Gx0) * (iz - iz0)

    Gy00 = Gy000 + (Gy100-Gy000) * (ix - ix0)
    Gy01 = Gy001 + (Gy101-Gy001) * (ix - ix0)
    Gy10 = Gy010 + (Gy110-Gy010) * (ix - ix0)
    Gy11 = Gy011 + (Gy111-Gy011) * (ix - ix0)
    Gy0 = Gy00 + (Gy10 - Gy00) * (iy - iy0)
    Gy1 = Gy01 + (Gy11 - Gy01) * (iy - iy0)
    Gy = Gy0 + (Gy1 - Gy0) * (iz - iz0)

    Gz00 = Gz000 + (Gz100-Gz000) * (ix - ix0)
    Gz01 = Gz001 + (Gz101-Gz001) * (ix - ix0)
    Gz10 = Gz010 + (Gz110-Gz010) * (ix - ix0)
    Gz11 = Gz011 + (Gz111-Gz011) * (ix - ix0)
    Gz0 = Gz00 + (Gz10 - Gz00) * (iy - iy0)
    Gz1 = Gz01 + (Gz11 - Gz01) * (iy - iy0)
    Gz = Gz0 + (Gz1 - Gz0) * (iz - iz0)

    return(np.array((Gx, Gy, -Gz)))

def trace_ray_runge_kutta(u, vgrid, start, finish):
    h = 0.1
    dx, dy, dz = vgrid["dx"], vgrid["dy"], vgrid["dz"]
    grad0 = np.stack(np.gradient(u, dx, dy, dz), axis=3)
    grad = lambda xyz: interp_grad(grad0, vgrid, xyz)
    start = seispy.coords.as_geographic(start).to_cartesian().dot(vgrid["R"])
    finish = seispy.coords.as_geographic(finish).to_cartesian().dot(vgrid["R"])
    start[2] = seispy.constants.EARTH_RADIUS - start[2]
    finish[2] = seispy.constants.EARTH_RADIUS - finish[2]
    print("START:", start)
    print("FINISH:", finish)
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
        #print(ray[-1])
        if np.sqrt(np.sum(np.square(ray[-1]-start))) > \
                np.sqrt(np.sum(np.square(ray[-2]-start))):
            break
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

def plot_rays(u, vgrid, rays):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for ray in rays:
        ax.plot(ray[:,0], ray[:,1], ray[:,2])
    cc = vgrid["coords"]
    cb = ax.scatter(cc[...,0], cc[...,1], cc[...,2],
                    c=u,
                    cmap=plt.get_cmap("jet_r"),
                    alpha=0.2)
    ax.invert_zaxis()
    fig.colorbar(cb)
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
    ax.set_zlim((-10, 250))
    cb = ax.scatter(cc[...,0], cc[...,1], cc[...,2],
                    c=u,
                    cmap=plt.get_cmap("jet_r"))
    plt.show()

if __name__ == "__main__":
    main()
