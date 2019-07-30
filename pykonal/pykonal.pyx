# cython:    boundscheck=False
# cython:    cdivision=True
# cython:    language_level=3
# distutils: language=c++

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
    def __init__(self, coord_sys='cartesian'):
        '''
        Solves the Eikonal equation in 3D cartesian coordinates.
        '''
        self._ndim      = 3
        self._class     = str(self.__class__).strip('>\'').split('.')[-1]
        self._coord_sys = coord_sys
        self._vgrid     = GridND(ndim=self._ndim, coord_sys=self.coord_sys)

    @property
    def close(self):
        if not hasattr(self, '_close'):
            self._close = Heap(self.uu)
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
    def coord_sys(self):
        return (self._coord_sys)

    @coord_sys.setter
    def coord_sys(self, value):
        value = value.lower()
        if value not in ('cartesian', 'spherical'):
            raise (ValueError(f'Invalid coord_sys specification: {value}'))
        self._coord_sys = value


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
            if self.coord_sys == 'spherical':
                self._norm[..., 1] *= self.pgrid[..., 0]
                self._norm[..., 2] *= self.pgrid[..., 0] \
                    * np.sin(self.pgrid[..., 1])
        return (self._norm)

    @property
    def pgrid(self):
        if not hasattr(self, '_pgrid'):
            self._pgrid = GridND(ndim=self._ndim, coord_sys=self.vgrid.coord_sys)
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
        if self.coord_sys == 'spherical':
            return (self.src_loc)
        else:
            r = np.sqrt(np.sum(np.square(self.src_loc)))
            t = np.arccos(self.src_loc[2] / r)
            p = np.arctan2(self.src_loc[1], self.src_loc[0])
            return (np.array([r, t, p], dtype=DTYPE_REAL))

    @property
    def src_xyz(self):
        if self.coord_sys == 'cartesian':
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
            if np.any(np.isinf(vvp)):
                raise (ValueError('Velocity model corrupted on interpolationg.'))
            self._vvp = vvp
        return (self._vvp)


    def solve(self):
        self._update()


    def trace_ray(self, *args, method='euler', tolerance=1e-2):
        if method.upper() == 'EULER':
            return (self._trace_ray_euler(*args, tolerance=tolerance))
        else:
            raise (NotImplementedError('Only Euler integration is implemented yet'))


    def transfer_travel_times_from(self, old, origin, rotate=False, set_alive=False):
        '''
        Transfer the velocity model from old EikonalSolver to self
        :param pykonal.EikonalSolver old: The old EikonalSolver to transfer from.
        :param tuple old_origin: The coordinates of the origin of old w.r.t. to the self frame of reference.
        '''

        pgrid_new = self.pgrid.map_to(old.coord_sys, origin, rotate=rotate)
        if old.coord_sys == 'spherical' and old.pgrid.min_coords[2] >= 0:
            pgrid_new[...,2] = np.mod(pgrid_new[...,2], 2*np.pi)
        uui = return_nan_on_error(LinearInterpolator3D(old.pgrid, old.uu))

        shape = pgrid_new.shape
        for i1 in range(shape[0]):
            for i2 in range(shape[1]):
                for i3 in range(shape[2]):
                    idx = (i1, i2, i3)
                    u = uui(pgrid_new[idx])
                    if not np.isnan(u):
                        self.uu[idx]       = u
                        self.is_far[idx]   = False
                        self.is_alive[idx] = set_alive
                        self.close.push(*idx)


    def transfer_velocity_from(self, old, origin, rotate=False):
        '''
        Transfer the velocity model from old EikonalSolver to self
        :param pykonal.EikonalSolver old: The old EikonalSolver to transfer from.
        :param tuple old_origin: The coordinates of the origin of old w.r.t. to the self frame of reference.
        '''

        vgrid_new = self.vgrid.map_to(old.coord_sys, origin, rotate=rotate)
        if old.coord_sys == 'spherical' and old.vgrid.min_coords[2] >= 0:
            vgrid_new[...,2] = np.mod(vgrid_new[...,2], 2*np.pi)
        vvi = return_nan_on_error(LinearInterpolator3D(old.vgrid, old.vv))
        self.vv = np.apply_along_axis(vvi, -1, vgrid_new)


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
        cdef Index3D             idx
        cdef Py_ssize_t          i

        # Initialization
        if hasattr(self, '_vvp'):
            del(self._vvp)
        errors = update(
            self.uu,
            self.vvp,
            self.is_alive,
            self.close,
            self.is_far,
            self.pgrid.node_intervals,
            self.norm
        )

        self.errors = {'denominator': errors[0], 'determinant': errors[1]}

        # Clean-up
        del(self._vvp)


class GridND(object):
    def __init__(self, ndim=3, coord_sys='cartesian'):
        self._ndim      = ndim
        self._class     = str(self.__class__).strip('>\'').split('.')[-1]
        self._update    = True
        self._iax_null  = None
        self._coord_sys = coord_sys


    @property
    def iax_null(self):
        return (self._iax_null)

    @iax_null.setter
    def iax_null(self, value):
        self._iax_null = value

    @property
    def coord_sys(self):
        return (self._coord_sys)

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


    def map_to(self, coord_sys, origin, rotate=False):
        '''
        Return the coordinates of self in the new reference frame.

        :param pykonal.GridND self: Coordinate grid to transform.
        :param str coord_sys: Coordinate system to transform to.
        :param origin: Coordinates of the origin of self w.r.t. the new frame of reference.
        '''
        origin = np.array(origin)
        if self.coord_sys == 'spherical' and coord_sys.lower() == 'spherical':
            xx_old = self[...,0] * np.sin(self[...,1]) * np.cos(self[...,2])
            yy_old = self[...,0] * np.sin(self[...,1]) * np.sin(self[...,2])
            zz_old = self[...,0] * np.cos(self[...,1])
            origin_xyz = [
                origin[0] * np.sin(origin[1]) * np.cos(origin[2]),
                origin[0] * np.sin(origin[1]) * np.sin(origin[2]),
                origin[0] * np.cos(origin[1])
            ]
            xx_new  = xx_old + origin_xyz[0]
            yy_new  = yy_old + origin_xyz[1]
            zz_new  = zz_old + origin_xyz[2]
            xyz_new = np.moveaxis(np.stack([xx_new,yy_new,zz_new]), 0, -1)

            rr_new             = np.sqrt(np.sum(np.square(xyz_new), axis=-1))
            old_error_settings = np.seterr(divide='ignore', invalid='ignore')
            tt_new             = np.arccos(xyz_new[...,2] / rr_new)
            np.seterr(**old_error_settings)
            pp_new             = np.arctan2(xyz_new[...,1], xyz_new[...,0])
            rtp_new            = np.moveaxis(
                np.stack([rr_new, tt_new, pp_new]),
                0,
                -1
            )
            return (rtp_new)
        elif self.coord_sys == 'cartesian' and coord_sys.lower() == 'spherical':
            origin_xyz = [
                origin[0] * np.sin(origin[1]) * np.cos(origin[2]),
                origin[0] * np.sin(origin[1]) * np.sin(origin[2]),
                origin[0] * np.cos(origin[1])
            ]
            if rotate is True:
                xyz_old = self[...].dot(
                    rotation_matrix(np.pi/2-origin[2], 0, np.pi/2-origin[1])
                )
            else:
                xyz_old = self[...]
            xyz_new            = xyz_old + origin_xyz
            rr_new             = np.sqrt(np.sum(np.square(xyz_new), axis=-1))
            old_error_settings = np.seterr(divide='ignore', invalid='ignore')
            tt_new             = np.arccos(xyz_new[...,2] / rr_new)
            np.seterr(**old_error_settings)
            pp_new      = np.arctan2(xyz_new[...,1], xyz_new[...,0])
            rtp_new = np.moveaxis(np.stack([rr_new,tt_new, pp_new]), 0, -1)
            return (rtp_new)
        elif self.coord_sys == 'spherical' and coord_sys.lower() == 'cartesian':
            origin_xyz = origin
            xx_old     = self[...,0] * np.sin(self[...,1]) * np.cos(self[...,2])
            yy_old     = self[...,0] * np.sin(self[...,1]) * np.sin(self[...,2])
            zz_old     = self[...,0] * np.cos(self[...,1])
            xx_new     = xx_old + origin_xyz[0]
            yy_new     = yy_old + origin_xyz[1]
            zz_new     = zz_old + origin_xyz[2]
            xyz_new    = np.moveaxis(np.stack([xx_new,yy_new,zz_new]), 0, -1)
            return (xyz_new)
        elif self.coord_sys == 'cartesian' and coord_sys.lower() == 'cartesian':
            return (self[...] + origin)
        else:
            raise (NotImplementedError())


    def __getitem__(self, key):
        return (self.mesh[key])


cdef class Heap(object):
    cdef cpp_vector[Index3D] _keys
    cdef _REAL_t[:,:,:]      _values

    def __init__(self, values):
        self._values = values

    @property
    def values(self):
        return (np.asarray(self._values))

    @values.setter
    def values(self, values):
        self._values = values

    @property
    def keys(self):
        cdef Index3D idx
        output = []
        for i in range(self._keys.size()):
            idx = self._keys[i]
            output.append((idx.i1, idx.i2, idx.i3))
        return (output)

    @property
    def size(self):
        return (self._keys.size())


    cpdef tuple pop(Heap self):
        '''
        Pop the smallest item off the heap, maintaining the heap invariant.
        '''
        cdef Index3D last, idx_return

        last = self._keys.back()
        self._keys.pop_back()
        if self._keys.size() > 0:
            idx_return = self._keys[0]
            self._keys[0] = last
            self._sift_up(0)
            return ((idx_return.i1, idx_return.i2, idx_return.i3))
        return ((last.i1, last.i2, last.i3))

    cpdef void push(Heap self, Py_ssize_t i1, Py_ssize_t i2, Py_ssize_t i3):
        '''
        Push item onto heap, maintaining the heap invariant.
        '''
        cdef Index3D idx
        idx.i1, idx.i2, idx.i3 = i1, i2, i3
        self._keys.push_back(idx)
        self._sift_down(0, self._keys.size()-1)


    cpdef void sift_down(Heap self, Py_ssize_t j_start, Py_ssize_t j):
        '''
        Doc string
        '''
        cdef Py_ssize_t j_parent
        cdef Index3D    idx_new, idx_parent

        idx_new = self._keys[j]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while j > j_start:
            j_parent = (j - 1) >> 1
            idx_parent = self._keys[j_parent]
            if self._values[idx_new.i1, idx_new.i2, idx_new.i3] < self._values[idx_parent.i1, idx_parent.i2, idx_parent.i3]:
                self._keys[j] = idx_parent
                j = j_parent
                continue
            break
        self._keys[j] = idx_new

    cdef void _sift_down(Heap self, Py_ssize_t j_start, Py_ssize_t j):
        '''
        Doc string
        '''
        cdef Py_ssize_t j_parent
        cdef Index3D    idx_new, idx_parent

        idx_new = self._keys[j]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while j > j_start:
            j_parent = (j - 1) >> 1
            idx_parent = self._keys[j_parent]
            if self._values[idx_new.i1, idx_new.i2, idx_new.i3] < self._values[idx_parent.i1, idx_parent.i2, idx_parent.i3]:
                self._keys[j] = idx_parent
                j = j_parent
                continue
            break
        self._keys[j] = idx_new


    cpdef void _sift_up(Heap self, Py_ssize_t j_start):
        '''
        Doc string
        '''
        cdef Py_ssize_t j, j_child, j_end, j_right
        cdef Index3D idx_child, idx_right, idx_new

        j_end = self._keys.size()
        j = j_start
        idx_new = self._keys[j_start]
        # Bubble up the smaller child until hitting a leaf.
        j_child = 2 * j_start + 1 # leftmost child position
        while j_child < j_end:
            # Set childpos to index of smaller child.
            j_right = j_child + 1
            idx_child, idx_right = self._keys[j_child], self._keys[j_right]
            if j_right < j_end and not self._values[idx_child.i1, idx_child.i2, idx_child.i3] < self._values[idx_right.i1, idx_right.i2, idx_right.i3]:
                j_child = j_right
            # Move the smaller child up.
            self._keys[j] = self._keys[j_child]
            j = j_child
            j_child = 2 * j + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self._keys[j] = idx_new
        self._sift_down(j_start, j)


    cpdef Py_ssize_t which(Heap self, Py_ssize_t i1, Py_ssize_t i2, Py_ssize_t i3):
        cdef int     i
        cdef Index3D idx
        for i in range(self._keys.size()):
            idx = self._keys[i]
            if (idx.i1, idx.i2, idx.i3) == (i1, i2, i3):
                return (i)
        return (-1)

    cpdef to_list(self):
        cdef list    output
        cdef list    keys
        output = []
        keys   = self.keys

        for i in range(self._keys.size()):
            key = self.pop()
            output.append(key)
        for key in keys:
            self.push(*key)
        return (output)

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
            if (
                    point[iax] < self._min_coords[iax]
                    and not np.isclose(point[iax], self._min_coords[iax])

            ) or (
                    point[iax] > self._max_coords[iax]
                    and not np.isclose(point[iax], self._max_coords[iax])
            ):
                raise(
                    OutOfBoundsError(
                        f'Point outside of interpolation domain requested: ({point[0]}, {point[1]}, {point[2]})'
                    )
                )
            idx[iax] = (point[iax] - self._min_coords[iax]) / self._node_intervals[iax]
            delta[iax] = (idx[iax] % 1.) * self._node_intervals[iax]
        i1   = <Py_ssize_t> idx[0]
        i2   = <Py_ssize_t> idx[1]
        i3   = <Py_ssize_t> idx[2]
        di1  = 0 if self._iax_isnull[0] == 1 or i1 == self._max_idx[0] else 1
        di2  = 0 if self._iax_isnull[1] == 1 or i2 == self._max_idx[1] else 1
        di3  = 0 if self._iax_isnull[2] == 1 or i3 == self._max_idx[2] else 1
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


def return_nan_on_error(func):
    def wrapper(*args):
        try:
            return (func(*args))
        except Exception:
            return (np.nan)
    return (wrapper)

def rotation_matrix(alpha, beta, gamma):
    '''
    Return the rotation matrix used to rotate a set of cartesian
    coordinates by alpha radians about the z-axis, then beta radians
    about the y'-axis and then gamma radians about the z''-axis.
    '''
    aa = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha),  np.cos(alpha), 0],
            [0,              0,             1]
        ]
    )
    bb = np.array(
        [
            [ np.cos(beta), 0, np.sin(beta)],
            [ 0,            1, 0           ],
            [-np.sin(beta), 0, np.cos(beta)]
        ]
    )
    cc = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma),  np.cos(gamma), 0],
            [0,               0,            1]
        ]
    )
    return (aa.dot(bb).dot(cc))


cdef Index3D heap_pop(cpp_vector[Index3D]& idxs, _REAL_t[:,:,:] uu):
    '''
    Pop the smallest item off the heap, maintaining the heap invariant.
    '''
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
    '''
    Push item onto heap, maintaining the heap invariant.
    '''
    idxs.push_back(idx)
    sift_down(idxs, uu, 0, idxs.size()-1)


cdef void sift_down(
    cpp_vector[Index3D]& idxs,
    _REAL_t[:,:,:] uu,
    Py_ssize_t j_start,
    Py_ssize_t j
):
    '''
    Doc string
    '''
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
    '''
    Doc string
    '''
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


cdef void heapify(
    cpp_vector[Index3D]& idxs,
    _REAL_t[:,:,:] uu
):
    for j_start in reversed(range(idxs.size()//2)):
        sift_up(idxs, uu, j_start)

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
        Heap close,
        np.ndarray[np.npy_bool, ndim=3, cast=True] is_far,
        _REAL_t[:] dd,
        _REAL_t[:,:,:,:] norm
):
    '''
    The update algorithm to propagate the wavefront.
    '''
    cdef Py_ssize_t       i, iax, idrxn, trial_i1, trial_i2, trial_i3
    cdef Py_ssize_t[6][3] nbrs
    cdef Py_ssize_t[3]    trial_idx, max_idx, nbr, switch
    cdef Py_ssize_t[2]    drxns = [-1, 1]
    cdef Index3D          idx
    cdef int              count_a = 0
    cdef int              count_b = 0
    cdef int              inbr
    cdef int[2]           order
    cdef int[3]           is_periodic
    cdef _REAL_t          a, b, c, bfd, ffd, new
    cdef _REAL_t[2]       fdu
    cdef _REAL_t[3]       aa, bb, cc

    is_periodic   = False, False, True
    max_idx       = [is_alive.shape[0], is_alive.shape[1], is_alive.shape[2]]

    while close.size > 0:
        # Let Trial be the point in Close with the smallest value of u
        trial_i1, trial_i2, trial_i3 = close.pop()
        trial_idx = [trial_i1, trial_i2, trial_i3]
        is_alive[trial_i1, trial_i2, trial_i3] = True

        # Determine the indices of neighbouring nodes.
        inbr = 0
        for iax in range(3):
            switch = [0, 0, 0]
            for idrxn in range(2):
                switch[iax] = drxns[idrxn]
                for jax in range(3):
                    nbrs[inbr][jax] = (
                          trial_idx[jax]
                        + switch[jax]
                        + (max_idx[jax] + 1) * is_periodic[jax]
                    )\
                    % (max_idx[jax] + 1)
                inbr += 1

        # Recompute the values of u at all Close neighbours of Trial
        # by solving the piecewise quadratic equation.
        for i in range(6):
            nbr_i1 = nbrs[i][0]
            nbr_i2 = nbrs[i][1]
            nbr_i3 = nbrs[i][2]
            nbr    = nbrs[i]
            if not stencil(nbr[0], nbr[1], nbr[2], max_idx[0], max_idx[1], max_idx[2]) \
                    or is_alive[nbr[0], nbr[1], nbr[2]]:
                continue
            if vv[nbr[0], nbr[1], nbr[2]] > 0 \
                    and not np.isnan(vv[nbr[0], nbr[1], nbr[2]]):
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
                    count_b += 1
                    continue
                else:
                    new = (-b + libc.math.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
                if new < uu[nbr[0], nbr[1], nbr[2]]:
                    uu[nbr[0], nbr[1], nbr[2]] = new
                    close._sift_down(0, close.which(*nbr))#nbr[0], nbr[1], nbr[2]))
                    # Tag as Close all neighbours of Trial that are not
                    # Alive. If the neighbour is in Far, remove it from
                    # that list and add it to Close.
                    if is_far[nbr[0], nbr[1], nbr[2]]:
                        close.push(nbr[0], nbr[1], nbr[2])
                        is_far[nbr[0], nbr[1], nbr[2]] = False
    return (count_a, count_b)
