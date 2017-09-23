import heapq
import numpy as np

Nx, Ny = 50, 50

def main():
    u = np.ones((Nx,Ny)) * float('inf')
    #v = np.ones((Nx,Ny)) * range(1, Nx+1)
    #v = v.T
    v = np.ones((Nx,Ny))
    v *= range(Ny, 0, -1)
    v /= Ny
    v *= 6
    u = np.ma.masked_array(u, mask=False)
    #start = (0,49)
    #u[start] = 0
    #u.mask[start] = True
    #live = []
    #heapq.heapify(live)
    #heapq.heappush(live, (0, start))
#####
    live = []
    heapq.heapify(live)
    start = [(12, 23)]
    for s in start:
        u[s] = 0
        u.mask[s] = True
        heapq.heappush(live, (0, s))
#####
    while len(live) > 0:
        u, live = update(u, v, live)
    u = np.ma.getdata(u)
    ray = trace_ray(u, start[0], (Nx-5, Ny-1))
    plot_ray(v, ray)
    #plot_ray(np.ma.getdata(u), ray)
    #plot(np.ma.getdata(u))

def update(u, v, live):
    h = 1
    u0 = np.ma.getdata(u)
    _, active = heapq.heappop(live)
    near = [(i, j) for (i, j) in [(active[0]-1, active[1]),
                                  (active[0]+1, active[1]),
                                  (active[0], active[1]-1),
                                  (active[0], active[1]+1)]
                   if 0 <= i < u0.shape[0]
                   and 0 <= j < u0.shape[1]
                   and not u.mask[i, j]]
    for (i, j) in near:
        if not (0 <= i < u0.shape[0] and 0 <= j < u0.shape[1]) or u.mask[i, j]:
            continue
        uh = min(u0[max(i-1, 0), j],
                 u0[min(i+1, u0.shape[0]-1), j])
        uv = min(u0[i, max(j-1, 0)],
                 u0[i, min(j+1, u0.shape[1]-1)])
        u[i, j] = 1/2 * (uh + uv + np.sqrt((uh + uv)**2 - \
                                           2*(uh**2 + uv**2 - (h/v[i, j])**2)))\
                if abs(uh - uv) <= h/v[i, j]\
                else min(uh, uv) + h/v[i,j]
    u.mask[active] = True
    indices = [l[1] for l in live]
    for ij in near:
        if ij in indices:
            index = indices.index(ij)
            live[index] = (u[ij], ij)
        else:
            heapq.heappush(live, (u[ij], ij))
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

def plot_ray(u, ray):
    import matplotlib.pyplot as plt
    X, Y = np.meshgrid(range(Nx), range(Ny), indexing="ij")
    plt.plot(ray[:,0], ray[:,1])
    plt.scatter(X, Y, s=5, c=u, cmap=plt.get_cmap("jet"))
    plt.colorbar()
    plt.show()

def plot(u):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    pointsize = 5
    grad = np.gradient(np.ma.getdata(u))
    grad[0] *= -1
    grad[1] *= -1
    X, Y = np.meshgrid(range(Nx), range(Ny), indexing="ij")
    fig = plt.figure()
    ax = fig.add_subplot(1, 3, 1)
    #ax.scatter(X, Y, s=pointsize, c=u, cmap=plt.get_cmap("jet"))
    ax.quiver(X, Y, grad[0], grad[1], angles="xy", scale_units="xy", scale=10)
    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(X, Y,
               c=np.stack((grad[0], grad[1]), axis=2)[...,0],
               s=pointsize,
               cmap=plt.get_cmap("jet"))
    ax = fig.add_subplot(1, 3, 3)
    cb = ax.scatter(X, Y,
                    c=np.stack((grad[0], grad[1]), axis=2)[...,1],
                    s=pointsize,
                    cmap=plt.get_cmap("jet"))
    fig.colorbar(cb)
    plt.show()

if __name__ == "__main__":
    main()
