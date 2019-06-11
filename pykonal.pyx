# cython: boundscheck=False,cdivision=True
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


class OutOfBoundsError(Exception):
    def __init__(self, msg=''):
        self.msg = msg


class EikonalSolver(object):
    def __init__(self, ndim=3):
        self._ndim    = ndim
        self._class   = str(self.__class__).strip('>\'').split('.')[-1]
        self._vgrid   = GridND(ndim=ndim)
        self._pgrid   = GridND(ndim=ndim)
        self._solved  = False
        self._sources = []


    @property
    def iax_null(self):
        return (self.pgrid.iax_null)

    @property
    def vgrid(self):
        return (self._vgrid)

    @property
    def ndim(self):
        return (self._ndim)


    @property
    def vv(self):
        return (self._vv)
    
    @vv.setter
    def vv(self, value):
        if not np.all(value.shape == self.vgrid.npts):
            raise (ValueError('SHAPE ERROR!'))
        self._vv = value
    
    @property
    def vv_p(self):
        if np.any(self.pgrid.min_coords < self.vgrid.min_coords) \
                or np.any(self.pgrid.max_coords > self.vgrid.max_coords):
            raise(OutOfBoundsError('Propagation grid extends beyond velocity grid boundaries. '\
                                   'Please re-initialize the propagation grid to lie entirely '\
                                   'within the velocity grid'))
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
        return(
            scipy.ndimage.map_coordinates(
                self.vv,
                [idx_new[iax].flatten() for iax in range(self.ndim)],
                output=np.float32
            ).reshape(
                self.pgrid.npts
            )
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
        self._sources.append((src, t0))
    

    @property
    def sources(self):
        sources = []
        for src, t0 in self._sources:
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
                t = t0 + np.sqrt(np.sum(np.square(self.pgrid[idx] - src))) / self.vv_p[idx]
                sources.append((idx, t))
        return (sources)


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
        
        init_sources(self.sources, uu, close, is_far)
        update(uu, self.vv_p, is_alive, close, is_far, self.pgrid.node_intervals)
        
        self._uu = np.array(uu)
        self._solved = True


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
        # Create an interpolator for the gradient field
        gg = np.moveaxis(np.stack(np.gradient(self.uu, axis=[iax for iax in range(self.ndim) if iax not in self.iax_null ])), 0, -1)
        for iax in self.iax_null:
            gg = np.insert(gg, iax, np.zeros(self.pgrid.npts), axis=-1)
        grad = LinearInterpolator3D(self.pgrid, gg)
        # Create an interpolator for the travel-time field
        uu = LinearInterpolator3D(self.pgrid, self.uu)
        ray = [np.array(start)]
        while uu(ray[-1]) > tolerance:
            ray.append(ray[-1] - step_size * grad(ray[-1]))
            if np.any(ray[-1] < self.pgrid.min_coords) or np.any(ray[-1] > self.pgrid.max_coords):
                raise (OutOfBoundsError('Ray went out of bounds'))
        return (np.array(ray))


class GridND(object):
    def __init__(self, ndim=3):
        self._ndim = ndim
        self._class = str(self.__class__).strip('>\'').split('.')[-1]
        self._update = True
        self._iax_null = None


    @property
    def iax_null(self):
        return (self._iax_null)

    @iax_null.setter
    def iax_null(self, value):
        self._iax_null = value
        

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
        self.iax_null = np.argwhere(value == 1).flatten()
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
        return ((self.min_coords + self.node_intervals * (self.npts - 1)).astype(np.float32))


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


class LinearInterpolator3D(object):
    def __init__(self, grid, values):
        self._grid = grid
        self._values = values
        self._max_idx = grid.npts - 1


    @property
    def grid(self):
        return (self._grid)

    @property
    def max_idx(self):
        return (self._max_idx)


    @property
    def values(self):
        return(self._values)        


    def __call__(self, point):
        point = np.array(point)
        dx, dy, dz = self.grid.node_intervals
        idx = (point - self.grid.min_coords) / self.grid.node_intervals
        for iax in range(3):
            if (idx[iax] < 0 and iax not in self.grid.iax_null) or idx[iax] > self.max_idx[iax]:
                raise(OutOfBoundsError('Point outside of interpolation domain requested'))
        delta_x, delta_y, delta_z = np.mod(idx, 1) * self.grid.node_intervals
        ix, iy, iz = idx.astype(np.int32)
        dix = 0 if 0 in self.grid.iax_null else 1
        diy = 0 if 1 in self.grid.iax_null else 1
        diz = 0 if 2 in self.grid.iax_null else 1
        f000 = self.values[ix,     iy,     iz]
        f100 = self.values[ix+dix, iy,     iz]
        f110 = self.values[ix+dix, iy+diy, iz]
        f101 = self.values[ix+dix, iy,     iz+diz]
        f111 = self.values[ix+dix, iy+diy, iz+diz]
        f010 = self.values[ix,     iy+diy, iz]
        f011 = self.values[ix,     iy+diy, iz+diz]
        f001 = self.values[ix,     iy,     iz+diz]
        f00  = f000 + (f100 - f000) / dx * delta_x
        f10  = f010 + (f110 - f010) / dx * delta_x
        f01  = f001 + (f101 - f001) / dx * delta_x
        f11  = f011 + (f111 - f011) / dx * delta_x
        f0   = f00  + (f10  - f00)  / dy * delta_y
        f1   = f01  + (f11  - f01)  / dy * delta_y
        f    = f0   + (f1   - f0)   / dz * delta_z
        return (f)


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
    '''The update algorithm to propagate the wavefront.'''
    cdef Py_ssize_t       i, iax, idrxn, trial_ix, trial_iy, trial_iz
    cdef Py_ssize_t[6][3] nbrs
    cdef Py_ssize_t[3]    max_idx, nbr, switch
    cdef Py_ssize_t[2]    drxns = [-1, 1]
    cdef Index3D          trial_idx, idx
    cdef int              count_a = 0
    cdef int              count_b = 0
    cdef int[2]           order
    cdef float            a, b, c, bfd, ffd
    cdef float[2]         fdu
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
                idrxn = 0
                for idrxn in range(2):
                    switch[iax] = drxns[idrxn]
                    if (
                               (drxns[idrxn] == -1 and nbr[iax] > 1) 
                            or (drxns[idrxn] == 1 and nbr[iax] < max_idx[iax] - 2)
                    )\
                            and is_alive[
                                nbr[0]+2*switch[0], 
                                nbr[1]+2*switch[1], 
                                nbr[2]+2*switch[2]
                            ]\
                            and is_alive[
                                nbr[0]+switch[0],
                                nbr[1]+switch[1],
                                nbr[2]+switch[2]
                            ]\
                            and uu[
                                nbr[0]+2*switch[0], 
                                nbr[1]+2*switch[1], 
                                nbr[2]+2*switch[2]
                            ] <= uu[
                                nbr[0]+switch[0], 
                                nbr[1]+switch[1], 
                                nbr[2]+switch[2]
                            ]\
                    :
                        order[idrxn] = 2
                        fdu[idrxn]  = drxns[idrxn] * (
                          - 3 * uu[
                              nbr[0],
                              nbr[1],
                              nbr[2]
                          ]\
                          + 4 * uu[
                              nbr[0]+switch[0],
                              nbr[1]+switch[1],
                              nbr[2]+switch[2]
                          ]\
                          -     uu[
                              nbr[0]+2*switch[0],
                              nbr[1]+2*switch[1],
                              nbr[2]+2*switch[2]
                          ]
                        ) / (2 * dd[iax])
                    elif (
                               (drxns[idrxn] == -1 and nbr[iax] > 0) 
                            or (drxns[idrxn] ==  1 and nbr[iax] < max_idx[iax] - 1)
                    )\
                            and is_alive[
                                nbr[0]+switch[0],
                                nbr[1]+switch[1],
                                nbr[2]+switch[2]
                            ]\
                    :
                        order[idrxn] = 1
                        fdu[idrxn] = drxns[idrxn] * (
                            uu[
                                nbr[0]+switch[0],
                                nbr[1]+switch[1],
                                nbr[2]+switch[2]
                            ]
                          - uu[nbr[0], nbr[1], nbr[2]]
                        ) / dd[iax]
                    else:
                        order[idrxn], fdu[idrxn] = 0, 0
                if fdu[0] > -fdu[1]:
                    # Do the update using the backward operator
                    idrxn, switch[iax] = 0, -1
                else:
                    # Do the update using the forward operator
                    idrxn, switch[iax] = 1, 1
                if order[idrxn] == 2:
                    aa[iax] = 9 / (4 * dd2[iax])
                    bb[iax] = (
                        6 * uu[
                            nbr[0]+2*switch[0],
                            nbr[1]+2*switch[1],
                            nbr[2]+2*switch[2]
                        ]
                     - 24 * uu[
                            nbr[0]+switch[0],
                            nbr[1]+switch[1],
                            nbr[2]+switch[2]
                        ]
                    ) / (4 * dd2[iax])
                    cc[iax] = (
                        uu[
                            nbr[0]+2*switch[0],
                            nbr[1]+2*switch[1],
                            nbr[2]+2*switch[2]
                        ]**2 \
                        - 8 * uu[
                            nbr[0]+2*switch[0],
                            nbr[1]+2*switch[1],
                            nbr[2]+2*switch[2]
                        ] * uu[
                            nbr[0]+switch[0],
                            nbr[1]+switch[1],
                            nbr[2]+switch[2]
                        ]
                        + 16 * uu[
                            nbr[0]+switch[0],
                            nbr[1]+switch[1],
                            nbr[2]+switch[2]
                        ]**2
                    ) / (4 * dd2[iax])
                elif order[idrxn] == 1:
                    aa[iax] = 1 / dd2[iax]
                    bb[iax] = -2 * uu[
                        nbr[0]+switch[0],
                        nbr[1]+switch[1],
                        nbr[2]+switch[2]
                    ] / dd2[iax]
                    cc[iax] = uu[
                        nbr[0]+switch[0],
                        nbr[1]+switch[1],
                        nbr[2]+switch[2]
                    ]**2 / dd2[iax]
                elif order[idrxn] == 0:
                    aa[iax], bb[iax], cc[iax] = 0, 0, 0
            a = aa[0] + aa[1] + aa[2]
            b = bb[0] + bb[1] + bb[2]
            c = cc[0] + cc[1] + cc[2] - 1/vv[nbr[0], nbr[1], nbr[2]]**2
            if a == 0:
                count_a += 1
                continue
            if b ** 2 < 4 * a * c:
                # This may not be mathematically permissible
                uu[nbr[0], nbr[1], nbr[2]] = -b / (2 * a)
                count_b += 1
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
    if count_a > 0:
        print(f'Denominator was zero {count_a} times')
    if count_b > 0:
        print(f'Determinant was negative {count_b} times')

