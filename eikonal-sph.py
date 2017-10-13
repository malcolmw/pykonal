import heapq
import numpy as np
import seispy
import time

pi = np.pi

VELOCITY_MODEL = "data/VpVs.dat"
SOURCE = (33.5, -116.5, 12.0)
OUTFILE = open("/Users/malcolcw/Desktop/pykonal.out", "w")

def main():
    vgrid = prep_vmod()
    sc, grid = vgrid["coords"], vgrid["grid"]
    #plot_vgrid(vgrid)
    u = np.ones(vgrid["velocity"].shape) * float('inf')
    u = np.ma.masked_array(u, mask=False)
################################################################################
# Initialize source.
    live = []
    heapq.heapify(live)
    source = seispy.coords.as_geographic(SOURCE).to_spherical()
    irho = (source[0] - grid.rho0) / (grid.drho)
    itheta = (source[1] - grid.theta0) / (grid.dtheta)
    iphi = (source[2] - grid.phi0) / (grid.dphi)
    irho0, irho1 = max(int(irho), 0), min(int(irho)+1, grid.nrho-1)
    itheta0, itheta1 = max(int(itheta), 0), min(int(itheta)+1, grid.ntheta-1)
    iphi0, iphi1 = max(int(iphi), 0), min(int(iphi)+1, grid.nphi-1)
    for i, j, k in [(irho0, itheta0, iphi0),
                    (irho0, itheta0, iphi1),
                    (irho0, itheta1, iphi0),
                    (irho0, itheta1, iphi1),
                    (irho1, itheta0, iphi0),
                    (irho1, itheta0, iphi1),
                    (irho1, itheta1, iphi0),
                    (irho1, itheta1, iphi1)]:
        u[i, j, k] = np.linalg.norm(source.to_cartesian() - sc[i, j, k].to_cartesian()) / vgrid["velocity"][i, j, k]
        u.mask[i, j, k] = True
        heapq.heappush(live, (u[i, j, k], (i, j, k)))
    live.sort()
################################################################################
# Solve eikonal equation
    t = time.time()
    print("Starting update")
    while len(live) > 0:
        u, live = update(u, vgrid, live)
    print("Update took %.3f s" % (time.time() - t))
    u = np.ma.getdata(u)
################################################################################
    ugrid = {"u": u, "grid": grid, "coords": sc, "gamma": vgrid["gamma"],
             "gammainv": vgrid["gammainv"]}
    finish = seispy.coords.as_geographic([34.0, -117.0, 0.0])
    rays = [trace_ray_runge_kutta(ugrid, source.to_geographic(), finish)]
    rays = [r.to_cartesian().dot(vgrid["gamma"]) for r in rays]
    plot_rays(u, vgrid, rays,
              source.to_cartesian().dot(vgrid["gamma"]),
              finish.to_cartesian().dot(vgrid["gamma"]))


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
        up, ddp2 = (up, 1/(rho*np.sin(theta)*dphi)**2) if up < u[i, j, k] else (0, 0)

        A = ddr2 + ddt2 + ddp2
        if A == 0:
            continue
        B = -2 * (ur*ddr2 + ut*ddt2 + up*ddp2)
        C = (ur**2)*ddr2 + (ut**2)*ddt2 + (up**2)*ddp2 - 1/v[i, j, k]**2

        #if B**2 < 4*A*C:
        #    print("A, B, C = %g, %g, %g" % (A, B, C), ddr2, ddt2, ddp2)
        #    print(ur, ut, up, u[i, j, k])
        #    print(active, (i, j, k))
        try:
            u[i, j, k] = (-B + max(0, np.sqrt(B**2 - 4*A*C))) / (2*A)
        except ZeroDivisionError:
            print(u[i, j, k], A, B, C, i, j, k)
            exit()
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
    gamma = seispy.coords.rotation_matrix(grid.phi0 + (grid.nphi - 1) * grid.dphi / 2,
                                          grid.theta0 + (grid.ntheta - 1) * grid.dtheta / 2,
                                          pi/2)
    vgrid = {"velocity": vm.v_type_grids[1][1]["data"],
             "coords": sc,
             "grid": grid,
             "gamma": gamma,
             "gammainv": np.linalg.inv(gamma)}
    return(vgrid)

def plot_vgrid(vgrid):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    cc = vgrid["coords"].to_cartesian().dot(vgrid["gamma"])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    cax = ax.scatter(cc[...,0], cc[...,1], cc[...,2],
                     s=1,
                     c=vgrid["velocity"],
                     cmap=plt.get_cmap("jet_r"))
    fig.colorbar(cax)
    plt.show()

def gradient(ugrid):
    grid = ugrid["grid"]
    sc = ugrid["coords"]
    u = ugrid["u"]
# Trim boundary layers
    sc = sc[1:-1, 1:-1, 1:-1]
# Central difference
    Gr = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2*grid.drho)
    Gt = (1 / sc[..., 0]) \
            * (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2 * grid.dtheta)
    Gp = (1 / (sc[..., 0] * np.sin(sc[...,1]))) \
            * (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * grid.dphi)
    lat0, lon0, depth0 = sc[-1, -1, 0].to_geographic()
    grid = seispy.geogrid.GeoGrid3D(lat0, lon0, depth0,
                                    grid.ntheta - 2,
                                    grid.nphi - 2,
                                    grid.nrho - 2,
                                    grid.dtheta * 180/pi,
                                    grid.dphi * 180/pi,
                                    grid.drho)
    gradient = {"G": np.stack((Gr, Gt, Gp), axis=3),
                "coords": sc,
                "grid": grid,
                "gamma": ugrid["gamma"],
                "gammainv": ugrid["gammainv"]}
    return(gradient)

def gradient_as_cartesian(gradient):
    G = gradient["G"]
    sc = gradient["coords"]
    Gx = G[..., 0] * np.sin(sc[..., 1]) * np.cos(sc[..., 2])\
            + G[..., 1] * np.cos(sc[..., 1]) * np.cos(sc[..., 2])\
            - G[..., 2] * np.sin(sc[..., 2])
    Gy = G[..., 0] * np.sin(sc[..., 1]) * np.sin(sc[..., 2])\
            + G[..., 1] * np.cos(sc[..., 1]) * np.sin(sc[..., 2])\
            + G[..., 2] * np.cos(sc[..., 2])
    Gz = G[..., 0] * np.cos(sc[...,1]) - G[..., 1] * np.sin(sc[..., 1])
    gradient = {"G": np.stack((Gx, Gy, Gz), axis=3),
                "coords": sc.to_cartesian(),
                "gamma": gradient["gamma"],
                "gammainv": gradient["gammainv"]}
    return(gradient)


def interpolate_gradient(gradient, xyz):
    rtp = seispy.coords.as_cartesian(xyz).dot(gradient["gammainv"]).to_spherical()
    grid = gradient["grid"]
    irho = (rtp[0] - grid.rho0) / grid.drho
    itheta = (rtp[1] - grid.theta0) / grid.dtheta
    iphi = (rtp[2] - grid.phi0) / grid.dphi
    irho0, irho1 = max(int(irho), 0), min(int(irho) + 1, grid.nrho - 1)
    itheta0, itheta1 = max(int(itheta), 0), min(int(itheta) + 1, grid.ntheta - 1)
    iphi0, iphi1 = max(int(iphi), 0), min(int(iphi) + 1, grid.nphi - 1)

    Gr000, Gt000, Gp000 = gradient["G"][irho0, itheta0, iphi0]
    Gr001, Gt001, Gp001 = gradient["G"][irho0, itheta0, iphi1]
    Gr010, Gt010, Gp010 = gradient["G"][irho0, itheta1, iphi0]
    Gr011, Gt011, Gp011 = gradient["G"][irho0, itheta1, iphi1]
    Gr100, Gt100, Gp100 = gradient["G"][irho1, itheta0, iphi0]
    Gr101, Gt101, Gp101 = gradient["G"][irho1, itheta0, iphi1]
    Gr110, Gt110, Gp110 = gradient["G"][irho1, itheta1, iphi0]
    Gr111, Gt111, Gp111 = gradient["G"][irho1, itheta1, iphi1]

    Gr00 = Gr000 + (Gr100-Gr000) * (irho - irho0)
    Gr01 = Gr001 + (Gr101-Gr001) * (irho - irho0)
    Gr10 = Gr010 + (Gr110-Gr010) * (irho - irho0)
    Gr11 = Gr011 + (Gr111-Gr011) * (irho - irho0)
    Gr0 = Gr00 + (Gr10 - Gr00) * (itheta - itheta0)
    Gr1 = Gr01 + (Gr11 - Gr01) * (itheta - itheta0)
    Gr = Gr0 + (Gr1 - Gr0) * (iphi - iphi0)

    Gt00 = Gt000 + (Gt100-Gt000) * (irho - irho0)
    Gt01 = Gt001 + (Gt101-Gt001) * (irho - irho0)
    Gt10 = Gt010 + (Gt110-Gt010) * (irho - irho0)
    Gt11 = Gt011 + (Gt111-Gt011) * (irho - irho0)
    Gt0 = Gt00 + (Gt10 - Gt00) * (itheta - itheta0)
    Gt1 = Gt01 + (Gt11 - Gt01) * (itheta - itheta0)
    Gt = Gt0 + (Gt1 - Gt0) * (iphi - iphi0)

    Gp00 = Gp000 + (Gp100-Gp000) * (irho - irho0)
    Gp01 = Gp001 + (Gp101-Gp001) * (irho - irho0)
    Gp10 = Gp010 + (Gp110-Gp010) * (irho - irho0)
    Gp11 = Gp011 + (Gp111-Gp011) * (irho - irho0)
    Gp0 = Gp00 + (Gp10 - Gp00) * (itheta - itheta0)
    Gp1 = Gp01 + (Gp11 - Gp01) * (itheta - itheta0)
    Gp = Gp0 + (Gp1 - Gp0) * (iphi - iphi0)

    Gx = Gr * np.sin(rtp[1]) * np.cos(rtp[2])\
            + Gt * np.cos(rtp[1]) * np.cos(rtp[2])\
            - Gp * np.sin(rtp[2])
    Gy = Gr * np.sin(rtp[1]) * np.sin(rtp[2])\
            + Gt * np.cos(rtp[1]) * np.sin(rtp[2])\
            + Gp * np.cos(rtp[2])
    Gz = Gr * np.cos(rtp[1]) - Gt * np.sin(rtp[1])

    return(seispy.coords.as_cartesian([Gx, Gy, Gz]).dot(gradient["gamma"]))

def trace_ray_runge_kutta2(u, vgrid, start, finish):
    h = 0.01
    grad0 = gradient(u, vgrid)
    grad = lambda xyz: interp_grad(grad0, vgrid, xyz)
    gamma = vgrid["gamma"]
    gammainv = vgrid["gammainv"]
    start = seispy.coords.as_geographic(start).to_cartesian().dot(gamma)
    finish = seispy.coords.as_geographic(finish).to_cartesian().dot(gamma)
    print("START:", start)
    print("FINISH:", finish)
    ray = np.array([finish])
    #for i in range(10000):
    while np.linalg.norm(ray[-1] - start) > h:
        print(ray[-1], start)
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
        print(ray[-1])
    return(seispy.coords.as_cartesian(ray).dot(gammainv).to_spherical())


def trace_ray_runge_kutta(ugrid, start, finish):
    h = 0.1
    grad0 = gradient(ugrid)
    print(grad0["grid"])
    grad = lambda xyz: interpolate_gradient(grad0, xyz)
    gamma = ugrid["gamma"]
    gammainv = ugrid["gammainv"]
    print("START:", start)
    print("FINISH:", finish)
    start = seispy.coords.as_geographic(start).to_cartesian().dot(gamma)
    finish = seispy.coords.as_geographic(finish).to_cartesian().dot(gamma)
    ray = np.array([finish])
    while len(ray) == 1 or np.linalg.norm(ray[-1]-start) < np.linalg.norm(ray[-2] - start):
        k1 = -grad(ray[-1])
        k2 = -grad(ray[-1] + h/2*k1)
        k3 = -grad(ray[-1] + h/2*k2)
        k4 = -grad(ray[-1] + h*k3)
        ray = np.vstack((ray,
                        ray[-1] + h/6*(k1 + 2*k2 + 2*k3 + k4)))
        print(ray[-1], np.linalg.norm(ray[-1]-start), np.linalg.norm(ray[-2] - start))
    return(seispy.coords.as_cartesian(ray).dot(gammainv).to_spherical())

def plot_rays(u, vgrid, rays, start, finish):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    for ray in rays:
        ax.plot(ray[:,0], ray[:,1], ray[:,2], "k")
    cc = vgrid["coords"].to_cartesian().dot(vgrid["gamma"])
    cb = ax.scatter(cc[...,0], cc[...,1], cc[...,2],
                    c=u,
                    cmap=plt.get_cmap("jet_r"),
                    alpha=0.05)
    ax.scatter(start[...,0], start[...,1], start[...,2], s=20, c="k")
    ax.scatter(finish[...,0], finish[...,1], finish[...,2], s=20, c="k")
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
    ax.set_zlim((-10, 250))
    cb = ax.scatter(cc[...,0], cc[...,1], cc[...,2],
                    c=u,
                    cmap=plt.get_cmap("jet_r"))
    plt.show()

if __name__ == "__main__":
    main()
