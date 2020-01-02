# Third-party imports
import numpy as np

# Local imports
from . import constants
from . import transform

# Third-party Cython imports
cimport numpy as np

# Local Cython imports
from . cimport constants


cdef class Field3D(object):

    def __init__(self, coord_sys='cartesian'):
        self._coord_sys   = coord_sys


    @property
    def coord_sys(self):
        return (self._coord_sys)

    @property
    def is_periodic(self):
        is_periodic = np.array([False, False, False])
        if self.coord_sys == 'spherical':
            is_periodic[2] = np.isclose(
                self.max_coords[2]+self.node_intervals[2]-self.min_coords[2],
                2 * np.pi
            )
        return (is_periodic)

    @property
    def node_intervals(self):
        return (np.asarray(self._node_intervals))

    @node_intervals.setter
    def node_intervals(self, value):
        self._node_intervals = np.array(value, dtype=constants.DTYPE_REAL)


    @property
    def npts(self):
        return (np.asarray(self.values.shape))


    @property
    def min_coords(self):
        return (np.asarray(self._min_coords))

    @min_coords.setter
    def min_coords(self, value):
        self._min_coords = np.array(value, dtype=constants.DTYPE_REAL)


    @property
    def values(self):
        return (None if self._values is None else np.asarray(self._values))

    @values.setter
    def values(self, value):
        self._values = np.array(value, dtype=constants.DTYPE_REAL)


    @property
    def max_coords(self):
        return ((self.min_coords + self.node_intervals * (self.npts - 1)))


    @property
    def nodes(self):
        nodes = [
            np.linspace(
                self.min_coords[idx],
                self.max_coords[idx],
                self.npts[idx],
                dtype=constants.DTYPE_REAL
            )
            for idx in range(3)
        ]
        nodes = np.meshgrid(*nodes, indexing='ij')
        nodes = np.stack(nodes)
        nodes = np.moveaxis(nodes, 0, -1)
        return (nodes)

    def map_to(self, coord_sys, origin, rotate=False):
        '''
        Return the coordinates of self in a new reference frame.
        :param coord_sys: Coordinate system to transform to ('*spherical*', or '*Cartesian*')
        :type coord_sys: str
        :param origin: Coordinates of the origin of self w.r.t. the new frame of reference.
        :type origin: 3-tuple, list, np.ndarray
        '''
        if self.coord_sys == 'spherical' and coord_sys.lower() == 'spherical':
            return (transform.sph2sph(self.nodes, origin))
        elif self.coord_sys == 'cartesian' and coord_sys.lower() == 'spherical':
            return (transform.xyz2sph(self.nodes, origin, rotate=rotate))
        elif self.coord_sys == 'spherical' and coord_sys.lower() == 'cartesian':
            return (transform.sph2xyz(self.nodes, origin))
        elif self.coord_sys == 'cartesian' and coord_sys.lower() == 'cartesian':
            return (transform.xyz2xyz(self.nodes, origin))
        else:
            raise (NotImplementedError())

    cpdef constants.REAL_t value(self, constants.REAL_t[:] point) except? -999999999999.:
        '''
        Interpolate the contained field at *point*.
        :param point: Coordinates of the point to interpolate at.
        :type point: np.ndarray[_REAL_t, ndim=1]
        :return: Value of the field at *point*.
        :rtype: REAL_t
        :raises: OutOfBoundsError
        '''
        cdef constants.REAL_t[3]         delta, idx
        cdef constants.REAL_t            f000, f100, f110, f101, f111, f010, f011, f001
        cdef constants.REAL_t            f00, f10, f01, f11
        cdef constants.REAL_t            f0, f1
        cdef constants.REAL_t            f
        cdef Py_ssize_t[3][2]            ii
        cdef Py_ssize_t                  i1, i2, i3, iax, di1, di2, di3

        for iax in range(3):
            #if (
            #    (
            #        point[iax] < self.min_coords[iax]
            #        or point[iax] > self.max_coords[iax]
            #    )
            #    and not self._is_periodic[iax]
            #    and not self._iax_isnull[iax]
            #):
            #    raise(
            #        OutOfBoundsError(
            #            f'Point outside of interpolation domain requested: ({point[0]}, {point[1]}, {point[2]})'
            #        )
            #    )
            idx[iax]   = (point[iax] - self.min_coords[iax]) / self.node_intervals[iax]
            #if self._iax_isnull[iax]:
            #    ii[iax][0] = 0
            #    ii[iax][1] = 0
            #else:
            ii[iax][0]  = <Py_ssize_t>idx[iax]
            ii[iax][1]  = <Py_ssize_t>(ii[iax][0]+1) % self.npts[iax]

            delta[iax] = idx[iax] % 1
        f000    = self.values[ii[0][0], ii[1][0], ii[2][0]]
        f100    = self.values[ii[0][1], ii[1][0], ii[2][0]]
        f110    = self.values[ii[0][1], ii[1][1], ii[2][0]]
        f101    = self.values[ii[0][1], ii[1][0], ii[2][1]]
        f111    = self.values[ii[0][1], ii[1][1], ii[2][1]]
        f010    = self.values[ii[0][0], ii[1][1], ii[2][0]]
        f011    = self.values[ii[0][0], ii[1][1], ii[2][1]]
        f001    = self.values[ii[0][0], ii[1][0], ii[2][1]]
        f00     = f000 + (f100 - f000) * delta[0]
        f10     = f010 + (f110 - f010) * delta[0]
        f01     = f001 + (f101 - f001) * delta[0]
        f11     = f011 + (f111 - f011) * delta[0]
        f0      = f00  + (f10  - f00)  * delta[1]
        f1      = f01  + (f11  - f01)  * delta[1]
        f       = f0   + (f1   - f0)   * delta[2]
        return (f)
