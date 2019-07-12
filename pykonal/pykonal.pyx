# cython: boundscheck=False
# cython: cdivision=True
# cython: language_level=3
# distutils: language = c++

# Python imports
import collections
import itertools
import numpy as np

# Cython imports
cimport numpy as np
cimport libc.math
from libcpp.vector cimport vector as cpp_vector
from libc.stdlib   cimport malloc, free

# Define the level of computational precision.
ctypedef np.float64_t _REAL_t
ctypedef np.uint16_t  _UINT_t
DTYPE_REAL = np.float64
DTYPE_UINT = np.uint16

DEF _ERROR_REAL = -999999999999.
ERROR_REAL      = DTYPE_REAL(_ERROR_REAL)

# A simple structure to hold 3D array indices.
cdef struct Index3D:
    Py_ssize_t i1, i2, i3

# A simple Exception class.
class OutOfBoundsError(Exception):
    def __init__(self, msg=''):
        self.msg = msg


class EikonalSolver(object):
    def __init__(self):
        '''
        Solves the Eikonal equation in 3D cartesian coordinates.
        '''
        self._ndim    = 3
        self._class   = str(self.__class__).strip('>\'').split('.')[-1]
        self._vgrid   = GridND(ndim=self._ndim)
        self._mode    = 'cartesian'

    @property
    def close(self):
        if not hasattr(self, '_close'):
            self._close = []
        return (self._close)

    @property
    def iax_null(self):
        return (self.pgrid.iax_null)

    @property
    def is_alive(self):
        if not hasattr(self, '_is_alive'):
            self._is_alive = np.full(self.pgrid.npts, fill_value=False, dtype=np.bool)
        return (self._is_alive)

    @property
    def is_far(self):
        if not hasattr(self, '_is_far'):
            self._is_far = np.full(self.pgrid.npts, fill_value=True, dtype=np.bool)
        return (self._is_far)

    @property
    def mode(self):
        return (self._mode)

    @mode.setter
    def mode(self, value):
        value = value.lower()
        if value not in ('hybrid-cartesian', 'hybrid-spherical', 'cartesian', 'spherical'):
            raise (ValueError(f'Invalid mode specification: {value}'))
        self._mode = value


    @property
    def ndim(self):
        return (self._ndim)

    @property
    def norm(self):
        if not hasattr(self, '_norm'):
            self._norm = np.tile(
                self.pgrid.node_intervals,
                np.append(self.pgrid.npts, 1)
            ).astype(DTYPE_REAL)
            if self.mode in ('spherical', 'hybrid-spherical'):
                self._norm[..., 1] *= self.pgrid[..., 0]
                self._norm[..., 2] *= self.pgrid[..., 0] \
                    * np.sin(self.pgrid[..., 1])
        return (self._norm)

    @property
    def pgrid(self):
        if not hasattr(self, '_pgrid'):
            self._pgrid = GridND(ndim=self._ndim)
            for attr in ('min_coords', 'node_intervals', 'npts'):
                setattr(self._pgrid, attr, getattr(self.vgrid, attr))
        return (self._pgrid)

    @property
    def src_loc(self):
        return (self._src_loc)

    @src_loc.setter
    def src_loc(self, value):
        if not isinstance(value, collections.Iterable):
            raise (TypeError(f'{self._class}.src_loc value must be <Iterable> type'))
        if len(value) != self._ndim:
            raise (ValueError(f'{self._class}.src_loc must have len() == {self._ndim}'))
        value = np.array(value, dtype=DTYPE_REAL)
        if np.any(value < self.pgrid.min_coords) or np.any(value > self.pgrid.max_coords):
            raise (OutOfBoundsError('Source location lies outside of propagation grid.'))
        self._src_loc = value

    @property
    def src_rtp(self):
        if self.mode in ('spherical', 'hybrid-spherical'):
            return (self.src_loc)
        else:
            r = np.sqrt(np.sum(np.square(self.src_loc)))
            t = np.arccos(self.src_loc[2] / r)
            p = np.arctan2(self.src_loc[1], self.src_loc[0])
            return (np.array([r, t, p], dtype=DTYPE_REAL))

    @property
    def src_xyz(self):
        if self.mode in ('cartesian', 'hybrid-cartesian'):
            return (self.src_loc)
        else:
            x = self.src_loc[0] * np.sin(self.src_loc[1]) * np.cos(self.src_loc[2])
            y = self.src_loc[0] * np.sin(self.src_loc[1]) * np.sin(self.src_loc[2])
            z = self.src_loc[0] * np.cos(self.src_loc[1])
            return (np.array([x, y, z], dtype=DTYPE_REAL))


    @property
    def uu(self):
        if not hasattr(self, '_uu'):
            self._uu = np.full(self.pgrid.npts, fill_value=np.inf, dtype=DTYPE_REAL)
        return (self._uu)

    @property
    def vgrid(self):
        return (self._vgrid)

    @property
    def vv(self):
        return (self._vv)

    @vv.setter
    def vv(self, value):
        if not np.all(value.shape == self.vgrid.npts):
            raise (ValueError('SHAPE ERROR!'))
        self._vv = value.astype(DTYPE_REAL)

    @property
    def vvp(self):
        cdef Py_ssize_t                i1, i2, i3
        cdef np.ndarray[_REAL_t, ndim=3] vvp

        if not hasattr(self, '_vvp'):
            if np.any(self.pgrid.min_coords < self.vgrid.min_coords) \
                    or np.any(self.pgrid.max_coords > self.vgrid.max_coords):
                raise(
                    OutOfBoundsError(
                        'Propagation grid extends beyond velocity grid '
                        'boundaries. Please re-initialize the propagation grid '
                        'to lie entirely within the velocity grid'
                    )
                )
            vvp  = np.zeros(self.pgrid.npts, dtype=DTYPE_REAL)
            vi    = LinearInterpolator3D(self.vgrid, self.vv).interpolate
            pgrid = self.pgrid[...]
            for i1 in range(self.pgrid.npts[0]):
                for i2 in range(self.pgrid.npts[1]):
                    for i3 in range(self.pgrid.npts[2]):
                        vvp[i1, i2, i3] = vi(pgrid[i1, i2, i3])
            if np.any(np.isnan(vvp))\
                    or np.any(np.isinf(vvp))\
            :
                raise (ValueError('Velocity model corrupted on interpolationg.'))
            self._vvp = vvp
        return (self._vvp)


    def solve(self):
        if self.mode in ('hybrid-cartesian', 'hybrid-spherical'):
            self._solve_hybrid()
        elif self.mode in ('cartesian', 'spherical'):
            self._update()
        else:
            raise (ValueError('Invalid mode specification. This should never happen.'))


    def trace_ray(self, *args, method='euler', tolerance=1e-2):
        if method.upper() == 'EULER':
            return (self._trace_ray_euler(*args, tolerance=tolerance))
        else:
            raise (NotImplementedError('Only Euler integration is implemented yet'))


    def _solve_hybrid(self):
        if not hasattr(self, '_src_loc'):
            raise (AttributeError('Source is unspecified.'))


        # Solve in the nearfield using spherical coordinates
        if self.mode == 'hybrid-cartesian':
            dr, nr            = np.min(self.pgrid.node_intervals), 10
            theta_min, ntheta = (0, 11) if 2 not in self.iax_null else (np.pi/2, 1)
            phi_min, nphi     = (0, 21) if 1 not in self.iax_null else (0, 1)
        elif self.mode == 'hybrid-spherical':
            dr, nr            = self.pgrid.node_intervals[0] / 5, 25
            theta_min, ntheta = (0, 11) if 1 not in self.iax_null else (np.pi/2, 1)
            phi_min, nphi     = (0, 21) if 2 not in self.iax_null else (0, 1)


        # Calculate the distance from the source to each node in the
        # farfield pgrid.
        if self.mode == 'hybrid-cartesian':
            dd = np.sqrt(np.sum(np.square(self.pgrid[...] - self.src_xyz), axis=-1))
        elif self.mode == 'hybrid-spherical':
            dd = np.sqrt(
                  np.square(self.pgrid[:,:,:,0])
                + np.square(self.src_rtp[0])
                - 2 * self.pgrid[:,:,:,0] * self.src_rtp[0] * (
                    np.sin(self.pgrid[:,:,:,1]) * np.sin(self.src_rtp[1]) * (
                          np.cos(self.pgrid[:,:,:,2])*np.cos(self.src_rtp[2])
                        + np.sin(self.pgrid[:,:,:,2])*np.sin(self.src_rtp[2])
                    )
                  + np.cos(self.pgrid[:,:,:,1]) * np.cos(self.src_rtp[1])
                )
            )
        # If the source lies directly on a node, the first layer of
        # nodes in the nearfield pgrid coincide with farfield nodes one
        # index away. Otherwise, the first layer of nodes coincides
        # with the nearest node.
        if np.any(dd == 0):
            r_min = dr
            for idx in np.argwhere(dd == 0):
                idx = tuple(idx)
                self.close.append(idx)
                self.uu[idx]     = 0
                self.is_far[idx] = False
        else:
            r_min = np.min(dd)


        self.near_field = near_field    = EikonalSolver()
        near_field.mode                 = 'spherical'
        near_field.vgrid.min_coords     = r_min, theta_min, phi_min
        near_field.vgrid.node_intervals = dr, np.pi/10, np.pi/10
        near_field.vgrid.npts           = nr, ntheta, nphi

        # Map the nearfield vgrid coordinates to the farfield
        # coordinate system.
        xx_near = near_field.vgrid[..., 0]\
            * np.sin(near_field.vgrid[..., 1])\
            * np.cos(near_field.vgrid[..., 2])\
            + self.src_xyz[0]
        yy_near = near_field.vgrid[..., 0]\
            * np.sin(near_field.vgrid[..., 1])\
            * np.sin(near_field.vgrid[..., 2])\
            + self.src_xyz[1]
        zz_near = near_field.vgrid[..., 0]\
            * np.cos(near_field.vgrid[..., 1])\
            + self.src_xyz[2]
        xyz_near = np.moveaxis(np.stack([xx_near,yy_near,zz_near]), 0, -1)

        if self.mode == 'hybrid-spherical':
            rr_near = np.sqrt(np.sum(np.square(xyz_near), axis=3))

            # Temporarily ignore divide by zero errors. A divide-by-zeros 
            # occurs when the source lies directly on a farfield pgrid
            # node.
            old = np.seterr(divide='ignore', invalid='ignore')
            tt_near = np.arccos(xyz_near[...,2] / rr_near)
            np.seterr(**old)

            pp_near = np.arctan2(xyz_near[...,1], xyz_near[...,0])

            icoords_near = np.moveaxis(np.stack([rr_near, tt_near, pp_near]), 0, -1)
        else:
            icoords_near = xyz_near

        def decorate(func, default):
            def wrapper(*args):
                try:
                    return (func(*args))
                except Exception:
                    return (default)
            return (wrapper)

        # Interpolate the velocity model onto the nearfield vgrid.
        vvi = decorate(LinearInterpolator3D(self.vgrid, self.vv).interpolate, 0)
        near_field.vv = np.apply_along_axis(vvi, -1, icoords_near)

        # Initialize the first layer of nearfield pgrid nodes.
        for it in range(near_field.pgrid.npts[1]):
            for ip in range(near_field.pgrid.npts[2]):
                idx = (0, it, ip)
                near_field.close.append(idx)
                near_field.is_far[idx]   = False
                near_field.is_alive[idx] = True
                if near_field.vv[idx] != 0:
                    near_field.uu[idx] = r_min / near_field.vv[idx]

        # Update the solver.
        near_field._update()

        # Transfer travel times from the nearfield grid to the farfield grid
        if self.mode == 'hybrid-cartesian':
            xyz_far = self.pgrid[...] - self.src_xyz
        elif self.mode == 'hybrid-spherical':
            xx_far = self.pgrid[..., 0]\
                * np.sin(self.pgrid[..., 1])\
                * np.cos(self.pgrid[..., 2])\
                - self.src_xyz[0]
            yy_far = self.pgrid[..., 0]\
                * np.sin(self.pgrid[..., 1])\
                * np.sin(self.pgrid[..., 2])\
                - self.src_xyz[1]
            zz_far = self.pgrid[..., 0]\
                * np.cos(self.pgrid[..., 1])\
                - self.src_xyz[2]
            xyz_far = np.moveaxis(np.stack([xx_far,yy_far,zz_far]), 0, -1)

        rr_far = np.sqrt(np.sum(np.square(xyz_far), axis=3))

        # Temporarily ignore divide by zero errors. A divide-by-zeros 
        # occurs when the source lies directly on a farfield pgrid
        # node.
        old = np.seterr(divide='ignore', invalid='ignore')
        tt_far = np.arccos(xyz_far[...,2] / rr_far)
        np.seterr(**old)

        pp_far = np.arctan2(xyz_far[...,1], xyz_far[...,0])

        rtp_far = np.moveaxis(np.stack([rr_far, tt_far, pp_far]), 0, -1)

        uui = decorate(
            LinearInterpolator3D(near_field.pgrid, near_field.uu).interpolate,
            np.nan
        )

        for idx in np.argwhere(
             (np.abs(rr_far) <= near_field.pgrid.max_coords[0])
            &(np.abs(rr_far) >= near_field.pgrid.min_coords[0])
        ):
            idx = tuple(idx)
            u = uui(rtp_far[idx])
            if not np.isnan(u):
                self.uu[idx] = u
                self.close.append(idx)
                self.is_far[idx]   = False
                self.is_alive[idx] = True

        self._update()


    def _trace_ray_euler(self, start):
        cdef cpp_vector[_REAL_t *]       ray
        cdef _REAL_t                     step_size, gx, gy, gz, norm
        cdef _REAL_t                     *point_new
        cdef _REAL_t[3]                  point_last, point_2last
        cdef Py_ssize_t                  i
        cdef np.ndarray[_REAL_t, ndim=2] ray_np

        point_new = <_REAL_t *> malloc(3 * sizeof(_REAL_t))
        point_new[0], point_new[1], point_new[2] = start
        ray.push_back(point_new)
        # step_size <-- half the smallest node_interval
        step_size = np.min(
            [
                self.pgrid.node_intervals[iax]
                for iax in range(self.ndim) if iax not in self.iax_null
            ]
        ) / 2
        # Create an interpolator for the gradient field
        gg = np.moveaxis(
            np.stack(
                np.gradient(
                    self.uu,
                    *[
                        self.pgrid.node_intervals[iax]
                        for iax in range(self.ndim) if iax not in self.iax_null
                    ],
                    axis=[
                        iax
                        for iax in range(self.ndim)
                        if iax not in self.iax_null
                    ]
                )
            ),
            0, -1
        )
        for iax in self.iax_null:
            gg = np.insert(gg, iax, np.zeros(self.pgrid.npts), axis=-1)
        grad_x = LinearInterpolator3D(self.pgrid, gg[...,0].astype(DTYPE_REAL))
        grad_y = LinearInterpolator3D(self.pgrid, gg[...,1].astype(DTYPE_REAL))
        grad_z = LinearInterpolator3D(self.pgrid, gg[...,2].astype(DTYPE_REAL))
        # Create an interpolator for the travel-time field
        uu = LinearInterpolator3D(self.pgrid, self.uu)
        point_last   = ray.back()
        while True:
            gx   = grad_x.interpolate(point_last)
            gy   = grad_y.interpolate(point_last)
            gz   = grad_z.interpolate(point_last)
            norm = libc.math.sqrt(gx**2 + gy**2 + gz**2)
            gx  /= norm
            gy  /= norm
            gz  /= norm
            point_new = <_REAL_t *> malloc(3 * sizeof(_REAL_t))
            point_new[0] = point_last[0] - step_size * gx
            point_new[1] = point_last[1] - step_size * gy
            point_new[2] = point_last[2] - step_size * gz
            point_2last = ray.back()
            ray.push_back(point_new)
            point_last   = ray.back()
            if uu.interpolate(point_2last) <= uu.interpolate(point_last):
                break
        ray_np = np.zeros((ray.size()-1, 3), dtype=DTYPE_REAL)
        for i in range(ray.size()-1):
            ray_np[i, 0] = ray[i][0]
            ray_np[i, 1] = ray[i][1]
            ray_np[i, 2] = ray[i][2]
            free(ray[i])
        return (ray_np)


    def _update(self):
        '''
        Update travel-time grid.
        '''
        cdef cpp_vector[Index3D] close
        cdef Index3D             idx
        cdef Py_ssize_t          i

        # Initialization
        for i1, i2, i3 in self.close:
            idx.i1, idx.i2, idx.i3 = i1, i2, i3
            heap_push(close, self.uu, idx)
        if hasattr(self, '_vvp'):
            del(self._vvp)

        errors = update(
            self.uu,
            self.vvp,
            self.is_alive,
            close,
            self.is_far,
            self.pgrid.node_intervals,
            self.norm
        )

        self.errors = {'denominator': errors[0], 'determinant': errors[1]}

        # Clean-up
        self._close = []
        del(self._vvp)


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
    def ndim(self):
        return (self._ndim)

    @property
    def node_intervals(self):
        return(self._node_intervals)

    @node_intervals.setter
    def node_intervals(self, value):
        if not isinstance(value, collections.Iterable):
            raise (TypeError(f'{self._class}.node_intervals value must be <Iterable> type'))
        if len(value) != self._ndim:
            raise (ValueError(f'{self._class}.node_intervals must have len() == {self._ndim}'))
        self._node_intervals = np.array(value, dtype=DTYPE_REAL)
        self._update = True


    @property
    def npts(self):
        return (self._npts)

    @npts.setter
    def npts(self, value):
        if not isinstance(value, collections.Iterable):
            raise (TypeError(f'{self._class}.delta value must be <Iterable> type'))
        if len(value) != self.ndim:
            raise (ValueError(f'{self._class}.delta must have len() == {self._ndim}'))
        self._npts = np.array(value, dtype=DTYPE_UINT)
        self.iax_null = np.argwhere(self.npts == 1).flatten()
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
        self._min_coords = np.array(value, dtype=DTYPE_REAL)
        self._update = True


    @property
    def max_coords(self):
        for attr in ('_node_intervals', '_npts', '_min_coords'):
            if not hasattr(self, attr):
                raise (AttributeError(f'{self._class}.{attr.lstrip("_")} not initialized'))
        return ((self.min_coords + self.node_intervals * (self.npts - 1)).astype(DTYPE_REAL))


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
            self._mesh = np.moveaxis(np.stack(mesh), 0, -1).astype(DTYPE_REAL)
            self._update = False
        return (self._mesh)


    def __getitem__(self, key):
        return (self.mesh[key])


cdef class LinearInterpolator3D(object):
    cdef _REAL_t[:,:,:,:] _grid
    cdef _REAL_t[:,:,:]   _values
    cdef _REAL_t[:]       _node_intervals
    cdef _REAL_t[3]       _min_coords
    cdef _REAL_t[3]       _max_coords
    cdef Py_ssize_t[3]    _max_idx
    cdef bint[3]          _iax_isnull

    def __init__(self, grid, values):
        self._grid           = grid[...]
        self._values         = values
        self._node_intervals = grid.node_intervals
        self._max_idx        = grid.npts - 1
        self._min_coords     = grid.min_coords
        self._max_coords     = grid.max_coords
        self._iax_isnull     = [
            True if iax in grid.iax_null else False for iax in range(grid.ndim)
        ]


    def __call__(self, point):
        return (self.interpolate(np.array(point, dtype=DTYPE_REAL)))


    cpdef _REAL_t interpolate(self, _REAL_t[:] point) except? _ERROR_REAL:
        cdef _REAL_t           f000, f100, f110, f101, f111, f010, f011, f001
        cdef _REAL_t           f00, f10, f01, f11
        cdef _REAL_t           f0, f1
        cdef _REAL_t           f
        cdef _REAL_t[3]        delta, idx
        cdef Py_ssize_t      i1, i2, i3, iax, di1, di2, di3

        for iax in range(3):
            if point[iax] < self._min_coords[iax] or point[iax] > self._max_coords[iax]:
                raise(
                    OutOfBoundsError(
                        f'Point outside of interpolation domain requested: ({point[0]}, {point[1]}, {point[2]})'
                    )
                )
            idx[iax] = (point[iax] - self._min_coords[iax]) / self._node_intervals[iax]
            delta[iax] = (idx[iax] % 1.) * self._node_intervals[iax]
        i1 = <Py_ssize_t> idx[0]
        i2 = <Py_ssize_t> idx[1]
        i3 = <Py_ssize_t> idx[2]
        di1 = 0 if self._iax_isnull[0] == 1 or i1 == self._max_idx[0] else 1
        di2 = 0 if self._iax_isnull[1] == 1 or i2 == self._max_idx[1] else 1
        di3 = 0 if self._iax_isnull[2] == 1 or i3 == self._max_idx[2] else 1
        f000 = self._values[i1,     i2,     i3]
        f100 = self._values[i1+di1, i2,     i3]
        f110 = self._values[i1+di1, i2+di2, i3]
        f101 = self._values[i1+di1, i2,     i3+di3]
        f111 = self._values[i1+di1, i2+di2, i3+di3]
        f010 = self._values[i1,     i2+di2, i3]
        f011 = self._values[i1,     i2+di2, i3+di3]
        f001 = self._values[i1,     i2,     i3+di3]
        f00  = f000 + (f100 - f000) / self._node_intervals[0] * delta[0]
        f10  = f010 + (f110 - f010) / self._node_intervals[0] * delta[0]
        f01  = f001 + (f101 - f001) / self._node_intervals[0] * delta[0]
        f11  = f011 + (f111 - f011) / self._node_intervals[0] * delta[0]
        f0   = f00  + (f10  - f00)  / self._node_intervals[1] * delta[1]
        f1   = f01  + (f11  - f01)  / self._node_intervals[1] * delta[1]
        f    = f0   + (f1   - f0)   / self._node_intervals[2] * delta[2]
        return (f)


cdef Index3D heap_pop(cpp_vector[Index3D]& idxs, _REAL_t[:,:,:] uu):
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


cdef void heap_push(cpp_vector[Index3D]& idxs, _REAL_t[:,:,:] uu, Index3D idx):
    '''Push item onto heap, maintaining the heap invariant.'''
    idxs.push_back(idx)
    sift_down(idxs, uu, 0, idxs.size()-1)


cdef void init_sources(
    list sources,
    _REAL_t[:,:,:] uu,
    cpp_vector[Index3D]& close,
    np.ndarray[np.npy_bool, ndim=3, cast=True] is_far
):
    cdef Index3D idx

    for source in sources:
        idx.i1, idx.i2, idx.i3 = source[0][0], source[0][1], source[0][2]
        uu[idx.i1, idx.i2, idx.i3] = source[1]
        is_far[idx.i1, idx.i2, idx.i3] = False
        heap_push(close, uu, idx)


cdef void sift_down(
    cpp_vector[Index3D]& idxs,
    _REAL_t[:,:,:] uu,
    Py_ssize_t j_start,
    Py_ssize_t j
):
    '''Doc string'''
    cdef Py_ssize_t j_parent
    cdef Index3D idx_new, idx_parent

    idx_new = idxs[j]
    # Follow the path to the root, moving parents down until finding a place
    # newitem fits.
    while j > j_start:
        j_parent = (j - 1) >> 1
        idx_parent = idxs[j_parent]
        if uu[idx_new.i1, idx_new.i2, idx_new.i3] < uu[idx_parent.i1, idx_parent.i2, idx_parent.i3]:
            idxs[j] = idx_parent
            j = j_parent
            continue
        break
    idxs[j] = idx_new


cdef void sift_up(
    cpp_vector[Index3D]& idxs,
    _REAL_t[:,:,:] uu,
    Py_ssize_t j_start
):
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
        if j_right < j_end and not uu[idx_child.i1, idx_child.i2, idx_child.i3] < uu[idx_right.i1, idx_right.i2, idx_right.i3]:
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
    Py_ssize_t i1,
    Py_ssize_t i2,
    Py_ssize_t i3,
    Py_ssize_t max_i1,
    Py_ssize_t max_i2,
    Py_ssize_t max_i3
):
    return (
            (i1 >= 0)
        and (i1 < max_i1)
        and (i2 >= 0)
        and (i2 < max_i2)
        and (i3 >= 0)
        and (i3 < max_i3)
    )


cdef tuple update(
        _REAL_t[:,:,:] uu,
        _REAL_t[:,:,:] vv,
        np.ndarray[np.npy_bool, ndim=3, cast=True] is_alive,
        cpp_vector[Index3D] &close,
        np.ndarray[np.npy_bool, ndim=3, cast=True] is_far,
        _REAL_t[:] dd,
        _REAL_t[:,:,:,:] norm
):
    '''The update algorithm to propagate the wavefront.'''
    cdef Py_ssize_t       i, iax, idrxn, trial_i1, trial_i2, trial_i3
    cdef Py_ssize_t[6][3] nbrs
    cdef Py_ssize_t[3]    max_idx, nbr, switch
    cdef Py_ssize_t[2]    drxns = [-1, 1]
    cdef Index3D          trial_idx, idx
    cdef int              count_a = 0
    cdef int              count_b = 0
    cdef int[2]           order
    cdef _REAL_t          a, b, c, bfd, ffd, new
    cdef _REAL_t[2]       fdu
    cdef _REAL_t[3]       aa, bb, cc


    max_idx       = [is_alive.shape[0], is_alive.shape[1], is_alive.shape[2]]

    while close.size() > 0:
        # Let Trial be the point in Close with the smallest value of u
        trial_idx = heap_pop(close, uu)
        trial_i1, trial_i2, trial_i3 = trial_idx.i1, trial_idx.i2, trial_idx.i3
        is_alive[trial_i1, trial_i2, trial_i3] = True

        nbrs[0][0] = trial_i1 - 1
        nbrs[0][1] = trial_i2
        nbrs[0][2] = trial_i3
        nbrs[1][0] = trial_i1 + 1
        nbrs[1][1] = trial_i2
        nbrs[1][2] = trial_i3
        nbrs[2][0] = trial_i1
        nbrs[2][1] = trial_i2 - 1
        nbrs[2][2] = trial_i3
        nbrs[3][0] = trial_i1
        nbrs[3][1] = trial_i2 + 1
        nbrs[3][2] = trial_i3
        nbrs[4][0] = trial_i1
        nbrs[4][1] = trial_i2
        nbrs[4][2] = trial_i3 - 1
        nbrs[5][0] = trial_i1
        nbrs[5][1] = trial_i2
        nbrs[5][2] = trial_i3 + 1
        for i in range(6):
            nbr_i1 = nbrs[i][0]
            nbr_i2 = nbrs[i][1]
            nbr_i3 = nbrs[i][2]
            nbr    = nbrs[i]
            if not stencil(nbr[0], nbr[1], nbr[2], max_idx[0], max_idx[1], max_idx[2]) \
                    or is_alive[nbr[0], nbr[1], nbr[2]]:
                continue
            # Recompute the values of u at all Close neighbours of Trial
            # by solving the piecewise quadratic equation.
            if vv[nbr[0], nbr[1], nbr[2]] > 0:
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
                            ) / (2 * norm[nbr[0], nbr[1], nbr[2], iax])
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
                            ) / norm[nbr[0], nbr[1], nbr[2], iax]
                        else:
                            order[idrxn], fdu[idrxn] = 0, 0
                    if fdu[0] > -fdu[1]:
                        # Do the update using the backward operator
                        idrxn, switch[iax] = 0, -1
                    else:
                        # Do the update using the forward operator
                        idrxn, switch[iax] = 1, 1
                    if order[idrxn] == 2:
                        aa[iax] = 9 / (4 * norm[nbr[0], nbr[1], nbr[2], iax] ** 2)
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
                        ) / (4 * norm[nbr[0], nbr[1], nbr[2], iax] ** 2)
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
                        ) / (4 * norm[nbr[0], nbr[1], nbr[2], iax] ** 2)
                    elif order[idrxn] == 1:
                        aa[iax] = 1 / norm[nbr[0], nbr[1], nbr[2], iax] ** 2
                        bb[iax] = -2 * uu[
                            nbr[0]+switch[0],
                            nbr[1]+switch[1],
                            nbr[2]+switch[2]
                        ] / norm[nbr[0], nbr[1], nbr[2], iax] ** 2
                        cc[iax] = uu[
                            nbr[0]+switch[0],
                            nbr[1]+switch[1],
                            nbr[2]+switch[2]
                        ]**2 / norm[nbr[0], nbr[1], nbr[2], iax] ** 2
                    elif order[idrxn] == 0:
                        aa[iax], bb[iax], cc[iax] = 0, 0, 0
                a = aa[0] + aa[1] + aa[2]
                if a == 0:
                    count_a += 1
                    continue
                b = bb[0] + bb[1] + bb[2]
                c = cc[0] + cc[1] + cc[2] - 1/vv[nbr[0], nbr[1], nbr[2]]**2
                if b ** 2 < 4 * a * c:
                    if -b / (2 * a) < uu[nbr[0], nbr[1], nbr[2]]:
                        # This may not be mathematically permissible
                        new = -b / (2 * a)
                    count_b += 1
                else:
                    new = (-b + libc.math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                if new < uu[nbr[0], nbr[1], nbr[2]]:
                    uu[nbr[0], nbr[1], nbr[2]] = new
            # Tag as Close all neighbours of Trial that are not Alive
            # If the neighbour is in Far, remove it from that list and add it to
            # Close
            if is_far[nbr[0], nbr[1], nbr[2]]:
                idx.i1, idx.i2, idx.i3 = nbr_i1, nbr_i2, nbr_i3
                heap_push(close, uu, idx)
                is_far[nbr[0], nbr[1], nbr[2]] = False
    return (count_a, count_b)
