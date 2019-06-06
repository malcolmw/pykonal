# distutils: language = c++

import collections
import itertools
import numpy as np
import scipy.interpolate
import scipy.ndimage
cimport numpy as np
cimport libc.math
from libcpp.vector cimport vector as cpp_vector


DTYPE = np.float32
cdef float MAX_FLOAT = np.finfo(DTYPE).max

cdef struct Index3D:
    Py_ssize_t ix, iy, iz


class EikonalSolver(object):
    def __init__(self, ndim=3):
        self._ndim    = ndim
        self._class   = str(self.__class__).strip('>\'').split('.')[-1]
        self._vgrid   = GridND(ndim=ndim)
        self._pgrid   = GridND(ndim=ndim)
        self._solved  = False
        self._sources = []


    @property
    def vgrid(self):
        return (self._vgrid)

    @property
    def ndim(self):
        return (self._ndim)


    @property
    def vv(self):
        return (self._vv.copy())
    
    @vv.setter
    def vv(self, value):
        if not np.all(value.shape == self.vgrid.npts):
            raise (ValueError('SHAPE ERROR!'))
        old_coords = np.meshgrid(
            *[
                np.linspace(
                    self.vgrid.min_coords[iax], 
                    self.vgrid.max_coords[iax], 
                    self.vgrid.npts[iax]
                ) 
                for iax in range(self.ndim)
            ],
            indexing='ij'
        )

        new_coords = np.meshgrid(
            *[
                np.linspace(
                    self.pgrid.min_coords[iax], 
                    self.pgrid.max_coords[iax], 
                    self.pgrid.npts[iax]
                ) 
                for iax in range(self.ndim)
            ],
            indexing='ij'
        )

        idx_start = [
            (new_coords[iax].min() - old_coords[iax].min()) / self.vgrid.node_intervals[iax]
            for iax in range(self.ndim)
        ]
        idx_end = [
            (new_coords[iax].max() - old_coords[iax].min()) / self.vgrid.node_intervals[iax]
            for iax in range(self.ndim)
        ]
        
        idx_new = np.meshgrid(
            *[
                np.linspace(
                    idx_start[iax], 
                    idx_end[iax], 
                    self.pgrid.npts[iax]
                ) 
                for iax in range(self.ndim)
            ],
            indexing='ij'
        )
        self._vv = scipy.ndimage.map_coordinates(
            value,
            [idx_new[iax].flatten() for iax in range(self.ndim)],
            output=np.float32
        ).reshape(
            self.pgrid.npts
        )


    @property
    def pgrid(self):
        return (self._pgrid)


    @property
    def uu(self):
        if self._solved is False:
            self.solve()
            self._solved = True
        return (self._uu)


    def add_source(self, src, t0=0):
        for iax in range(self.ndim):
            if self.pgrid.min_coords[iax] > src[iax] or self.pgrid.max_coords[iax] < src[iax]:
                raise (ValueError('Source location lies outside of propagation grid'))
        idx00 = (np.asarray(src) - self.pgrid.min_coords) / self.pgrid.node_intervals
        idx0 = idx00.astype(np.int32)
        mod = np.argwhere(np.mod(idx00, 1) != 0).flatten()
        idxs = []
        for delta in itertools.product(*[[0, 1] if idx in mod else [0] for idx in range(self.ndim)]):
            idxs.append(idx0 + np.array(delta))
        for idx in idxs:
            idx = tuple(idx)
            t = t0 + np.sqrt(np.sum(np.square(self.pgrid[idx] - src))) / self.vv[idx]
            self._sources.append((idx, t))


    def clear_sources(self):
        self._sources = []


    def solve(self):
        cdef cpp_vector[Index3D] close
        cdef float[:,:,:] uu
        cdef np.ndarray[np.npy_bool, ndim=3, cast=True] is_alive, is_far

        shape = self.pgrid.npts
        uu       = np.full(shape, fill_value=MAX_FLOAT, dtype=DTYPE)
        is_alive = np.full(shape, fill_value=False, dtype=np.bool)
        is_far   = np.full(shape, fill_value=True, dtype=np.bool)
        
        init_sources(self._sources, uu, close, is_far)
        update(uu, self.vv, is_alive, close, is_far, self.pgrid.node_intervals)
        
        self._uu = np.array(uu)


    def trace_ray(self, *args, method='euler', tolerance=1e-2):
        if method.upper() == 'EULER':
            return (self._trace_ray_euler(*args, tolerance=tolerance))
        else:
            raise (NotImplementedError('Only Euler integration is implemented yet'))


    def _trace_ray_euler(self, start, tolerance=1e-2):
        step_size = np.min(self.pgrid.node_intervals)
        # Create a flat array of coordinates
        coords = self.pgrid[...].reshape(
            np.prod(self.pgrid.npts), 
            self._ndim
        )
        # Create an interpolator for the Gradient field
        gg = scipy.interpolate.LinearNDInterpolator(
            coords, 
            np.stack([arr.flatten() for arr in np.gradient(self.uu)]).T
        )
        uu = scipy.interpolate.LinearNDInterpolator(
            coords, 
            self.uu.flatten()
        )
        ray = [np.array(start)]
        while uu(ray[-1]) > tolerance:
            ray.append(ray[-1] - step_size * gg(ray[-1])[0])
        return (np.array(ray))


class GridND(object):
    def __init__(self, ndim=3):
        self._ndim = ndim
        self._class = str(self.__class__).strip('>\'').split('.')[-1]
        self._update = True


    @property
    def node_intervals(self):
        return(self._node_intervals)
    
    @node_intervals.setter
    def node_intervals(self, value):
        if not isinstance(value, collections.Iterable):
            raise (TypeError(f'{self._class}.node_intervals value must be <Iterable> type'))
        if len(value) != self._ndim:
            raise (ValueError(f'{self._class}.node_intervals must have len() == {self._ndim}'))
        self._node_intervals = np.array(value, dtype=np.float32)
        self._update = True


    @property
    def npts(self):
        return (self._npts)
    
    @npts.setter
    def npts(self, value):
        if not isinstance(value, collections.Iterable):
            raise (TypeError(f'{self._class}.delta value must be <Iterable> type'))
        if len(value) != self._ndim:
            raise (ValueError(f'{self._class}.delta must have len() == {self._ndim}'))
        self._npts = np.array(value, dtype=np.int32)
        self._update = True


    @property
    def min_coords(self):
        return (self._min_coords)
    
    @min_coords.setter
    def min_coords(self, value):
        if not isinstance(value, collections.Iterable):
            raise (TypeError(f'{self._class}.min_coords value must be <Iterable> type'))
        if len(value) != self._ndim:
            raise (ValueError(f'{self._class}.min_coords must have len() == {self._ndim}'))
        self._min_coords = np.array(value, dtype=np.float32)
        self._update = True


    @property
    def max_coords(self):
        for attr in ('_node_intervals', '_npts', '_min_coords'):
            if not hasattr(self, attr):
                raise (AttributeError(f'{self._class}.{attr.lstrip("_")} not initialized'))
        return (self.min_coords + self.node_intervals * (self.npts - 1))


    @property
    def mesh(self):
        '''
        mesh is indexed like [iww0, iww1, ..., iwwN, iax]
        '''
        if self._update is True:
            mesh = np.meshgrid(
                *[
                    np.linspace(
                        self.min_coords[idx], 
                        self.max_coords[idx],
                        self.npts[idx]
                    )
                    for idx in range(self._ndim)
                ], 
                indexing='ij'
            )
            self._mesh = np.moveaxis(np.stack(mesh), 0, -1)
            self._update = False
        return (self._mesh)


    def __getitem__(self, key):
        return (self.mesh[key])


cdef Index3D heap_pop(cpp_vector[Index3D]& idxs, float[:,:,:] uu):
    '''Pop the smallest item off the heap, maintaining the heap invariant.'''
    cdef Index3D last, idx_return
    
    last = idxs.back()
    idxs.pop_back()
    if idxs.size() > 0:
        idx_return = idxs[0]
        idxs[0] = last
        sift_up(idxs, uu, 0)
        return (idx_return)
    return (last)


cdef void heap_push(cpp_vector[Index3D]& idxs, float[:,:,:] uu, Index3D idx):
    '''Push item onto heap, maintaining the heap invariant.'''
    idxs.push_back(idx)
    sift_down(idxs, uu, 0, idxs.size()-1)


cdef void init_sources(
    list sources,
    float[:,:,:] uu, 
    cpp_vector[Index3D]& close, 
    np.ndarray[np.npy_bool, ndim=3, cast=True] is_far
):
    cdef Index3D idx
    
    for source in sources:
        idx.ix, idx.iy, idx.iz = source[0][0], source[0][1], source[0][2]
        uu[idx.ix, idx.iy, idx.iz] = source[1]
        is_far[idx.ix, idx.iy, idx.iz] = False
        heap_push(close, uu, idx)


cdef void sift_down(cpp_vector[Index3D]& idxs, float[:,:,:] uu, Py_ssize_t j_start, Py_ssize_t j):
    '''Doc string'''
    cdef Py_ssize_t j_parent
    cdef Index3D idx_new, idx_parent

    idx_new = idxs[j]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while j > j_start:
        j_parent = (j - 1) >> 1
        idx_parent = idxs[j_parent]
        if uu[idx_new.ix, idx_new.iy, idx_new.iz] < uu[idx_parent.ix, idx_parent.iy, idx_parent.iz]:
            idxs[j] = idx_parent
            j = j_parent
            continue
        break
    idxs[j] = idx_new


cdef void sift_up(cpp_vector[Index3D]& idxs, float[:,:,:] uu, Py_ssize_t j_start):
    '''Doc string'''
    cdef Py_ssize_t j, j_child, j_end, j_right
    cdef Index3D idx_child, idx_right, idx_new
    
    j_end = idxs.size()
    j = j_start
    idx_new = idxs[j_start]
    # Bubble up the smaller child until hitting a leaf.
    j_child = 2 * j_start + 1 # leftmost child position
    while j_child < j_end:
        # Set childpos to index of smaller child.
        j_right = j_child + 1
        idx_child, idx_right = idxs[j_child], idxs[j_right]
        if j_right < j_end and not uu[idx_child.ix, idx_child.iy, idx_child.iz] < uu[idx_right.ix, idx_right.iy, idx_right.iz]:
            j_child = j_right
        # Move the smaller child up.
        idxs[j] = idxs[j_child]
        j = j_child
        j_child = 2 * j + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    idxs[j] = idx_new
    sift_down(idxs, uu, j_start, j)

cdef bint stencil(
        Py_ssize_t ix, Py_ssize_t iy, Py_ssize_t iz, Py_ssize_t max_ix, Py_ssize_t max_iy, Py_ssize_t max_iz
):
    return (
            (ix >= 0)
        and (ix < max_ix)
        and (iy >= 0)
        and (iy < max_iy)
        and (iz >= 0)
        and (iz < max_iz)
    )


cdef void update(
        float[:,:,:] uu,
        float[:,:,:] vv,
        np.ndarray[np.npy_bool, ndim=3, cast=True] is_alive,
        cpp_vector[Index3D] close,
        np.ndarray[np.npy_bool, ndim=3, cast=True] is_far,
        float[:] dd
):
    '''The update algorithm to propagate the wavefront. '''
#     cdef Py_ssize_t       *trial_idx
    cdef Py_ssize_t       i, iax, trial_ix, trial_iy, trial_iz
    cdef Py_ssize_t[6][3] nbrs
    cdef Py_ssize_t[3]    max_idx, nbr, switch
    cdef Index3D          trial_idx, idx
    cdef int              bord, ford, drxn
    cdef float            a, b, c, bfd, ffd
    cdef float[3]         aa, bb, cc, dd2

    max_idx       = [is_alive.shape[0], is_alive.shape[1], is_alive.shape[2]]
    dx, dy, dz    = dd
    dd2           = [dx**2, dy**2, dz**2]
    dx2, dy2, dz2 = dx**2, dy**2, dz**2
    for iax in range(3):
        assert dd[iax] > 0

    while close.size() > 0:
        # Let Trial be the point in Close with the smallest value of u
        trial_idx = heap_pop(close, uu)
        trial_ix, trial_iy, trial_iz = trial_idx.ix, trial_idx.iy, trial_idx.iz
        is_alive[trial_ix, trial_iy, trial_iz] = True

        nbrs[0][0] = trial_ix - 1
        nbrs[0][1] = trial_iy
        nbrs[0][2] = trial_iz
        nbrs[1][0] = trial_ix + 1
        nbrs[1][1] = trial_iy
        nbrs[1][2] = trial_iz
        nbrs[2][0] = trial_ix
        nbrs[2][1] = trial_iy - 1
        nbrs[2][2] = trial_iz
        nbrs[3][0] = trial_ix
        nbrs[3][1] = trial_iy + 1
        nbrs[3][2] = trial_iz
        nbrs[4][0] = trial_ix
        nbrs[4][1] = trial_iy
        nbrs[4][2] = trial_iz - 1
        nbrs[5][0] = trial_ix
        nbrs[5][1] = trial_iy
        nbrs[5][2] = trial_iz + 1
        for i in range(6):
            nbr_ix = nbrs[i][0]
            nbr_iy = nbrs[i][1]
            nbr_iz = nbrs[i][2]
            nbr    = nbrs[i]
            if not stencil(nbr[0], nbr[1], nbr[2], max_idx[0], max_idx[1], max_idx[2]) \
                    or is_alive[nbr[0], nbr[1], nbr[2]]:
                continue
            # Recompute the values of u at all Close neighbours of Trial
            # by solving the piecewise quadratic equation.
            for iax in range(3):
                switch = [0, 0, 0]
                switch[iax] = 1
                if nbr[iax] > 1 \
                        and not is_far[nbr[0]-switch[0]*2, nbr[1]-switch[1]*2, nbr[2]-switch[2]*2] \
                        and not is_far[nbr[0]-switch[0], nbr[1]-switch[1], nbr[2]-switch[2]] \
                        and uu[nbr[0]-switch[0]*2, nbr[1]-switch[1]*2, nbr[2]-switch[2]*2] <= uu[nbr[0]-switch[0], nbr[1]-switch[1], nbr[2]-switch[2]]:
                    bord = 2
                    bfd  = (
                        3 * uu[nbr[0],             nbr[1],             nbr[2]] \
                      - 4 * uu[nbr[0]-switch[0],   nbr[1]-switch[1],   nbr[2]-switch[2]] \
                      +     uu[nbr[0]-switch[0]*2, nbr[1]-switch[1]*2, nbr[2]-switch[2]*2]
                    ) / (2 * dd[iax])
                elif nbr[iax] > 0 \
                        and not is_far[nbr[0]-switch[0], nbr[1]-switch[1], nbr[2]-switch[2]]:
                    bord = 1
                    bfd  = (
                        uu[nbr[0],           nbr[1],           nbr[2]]
                      - uu[nbr[0]-switch[0], nbr[1]-switch[1], nbr[2]-switch[2]]
                    ) / dd[iax]
                else:
                    bfd, bord = 0, 0
                if nbr[iax] < max_idx[iax] - 2 \
                        and not is_far[nbr[0]+switch[0]*2, nbr[1]+switch[1]*2, nbr[2]+switch[2]*2] \
                        and not is_far[nbr[0]+switch[0],   nbr[1]+switch[1],   nbr[2]+switch[2]] \
                        and uu[nbr[0]+switch[0]*2, nbr[1]+switch[1]*2, nbr[2]+switch[2]*2] <= uu[nbr[0]+switch[0], nbr[1]+switch[1], nbr[2]+switch[2]]:
                    ford = 2
                    ffd  = (
                      - 3 * uu[nbr[0],             nbr[1],             nbr[2]] \
                      + 4 * uu[nbr[0]+switch[0],   nbr[1]+switch[1],   nbr[2]+switch[2]] \
                      -     uu[nbr[0]+switch[0]*2, nbr[1]+switch[1]*2, nbr[2]+switch[2]*2]
                    ) / (2 * dd[iax])
                elif nbr[iax] < max_idx[iax]-1 \
                        and not is_far[nbr[0]+switch[0], nbr[1]+switch[1], nbr[2]+switch[2]]:
                    ford = 1
                    ffd  = (
                        uu[nbr[0]+switch[0], nbr[1]+switch[1], nbr[2]+switch[2]]
                      - uu[nbr[0],           nbr[1],           nbr[2]]
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
                        6*uu[nbr[0]+2*drxn*switch[0], nbr[1]+2*drxn*switch[1], nbr[2]+2*drxn*switch[2]]
                     - 24*uu[nbr[0]+  drxn*switch[0], nbr[1]  +drxn*switch[1], nbr[2]  +drxn*switch[2]]
                    ) / (4 * dd2[iax])
                    cc[iax] = (
                        uu[
                            nbr[0]+2*drxn*switch[0],
                            nbr[1]+2*drxn*switch[1],
                            nbr[2]+2*drxn*switch[2]
                        ]**2 \
                        - 8 * uu[
                            nbr[0]+2*drxn*switch[0],
                            nbr[1]+2*drxn*switch[1],
                            nbr[2]+2*drxn*switch[2]
                        ] * uu[
                            nbr[0]+drxn*switch[0],
                            nbr[1]+drxn*switch[1],
                            nbr[2]+drxn*switch[2]
                        ]
                        + 16 * uu[
                            nbr[0]+drxn*switch[0],
                            nbr[1]+drxn*switch[1],
                            nbr[2]+drxn*switch[2]
                        ]**2
                    ) / (4 * dd2[iax])
                elif order == 1:
                    aa[iax] = 1 / dd2[iax]
                    bb[iax] = -2 * uu[
                        nbr[0]+drxn*switch[0],
                        nbr[1]+drxn*switch[1],
                        nbr[2]+drxn*switch[2]
                    ] / dd2[iax]
                    cc[iax] = uu[
                        nbr[0]+drxn*switch[0],
                        nbr[1]+drxn*switch[1],
                        nbr[2]+drxn*switch[2]
                    ]**2 / dd2[iax]
                elif order == 0:
                    aa[iax], bb[iax], cc[iax] = 0, 0, 0
                else:
                    raise (Exception('Huh!?'))
            a = aa[0] + aa[1] + aa[2]
            b = bb[0] + bb[1] + bb[2]
            c = cc[0] + cc[1] + cc[2] - 1/vv[nbr[0], nbr[1], nbr[2]]**2
            if a == 0:
#                 print(f'WARNING(2) :: a == 0 {nbr[0]}, {nbr[1]}')
                continue
            if b ** 2 < 4 * a * c:
                # This may not be mathematically permissible
                uu[nbr[0], nbr[1], nbr[2]] = -b / (2 * a)
#                 print(
#                     f'WARNING(2) :: determinant is negative {nbr[0]}, {nbr[1]}:'
#                     f'{100*np.sqrt(4 * a * c - b**2)/(2*a)/uu[nbr[0], nbr[1]]}'
#                 )
            else:
                uu[nbr[0], nbr[1], nbr[2]] = (
                    -b + libc.math.sqrt(b ** 2 - 4 * a * c)
                ) / (2 * a)
            # Tag as Close all neighbours of Trial that are not Alive
            # If the neighbour is in Far, remove it from that list and add it to
            # Close
            if is_far[nbr[0], nbr[1], nbr[2]]:
                idx.ix, idx.iy, idx.iz = nbr_ix, nbr_iy, nbr_iz
                heap_push(close, uu, idx)
                is_far[nbr[0], nbr[1], nbr[2]] = False
