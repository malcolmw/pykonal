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

    def __init__(self, coord_sys="cartesian"):
        self._coord_sys = coord_sys

    @property
    def coord_sys(self):
        return (self._coord_sys)

    @property
    def iax_isnull(self):
        return (self.npts == 1)

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
        self._node_intervals = np.asarray(value)


    @property
    def npts(self):
        return (np.asarray(self.values.shape))


    @property
    def min_coords(self):
        return (np.asarray(self._min_coords))

    @min_coords.setter
    def min_coords(self, value):
        self._min_coords = np.asarray(value)


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



cdef class ScalarField3D(Field3D):
    def __init__(self, coord_sys="cartesian"):
        super(ScalarField3D, self).__init__(coord_sys=coord_sys)

    @property
    def gradient(self):
        if self.coord_sys == "cartesian":
            return (self._get_gradient_cartesian())
        elif self.coord_sys == "spherical":
            return (self._get_gradient_spherical())


    @property
    def values(self):
        return (None if self._values is None else np.asarray(self._values))

    @values.setter
    def values(self, value):
        self._values = np.asarray(value)


    cpdef constants.REAL_t value(ScalarField3D self, constants.REAL_t[:] point) except? -999999999999.:
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
            if self.iax_isnull[iax]:
                ii[iax][0] = 0
                ii[iax][1] = 0
            else:
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


    #cpdef np.ndarray[constants.REAL_t, ndim=4] _get_gradient_cartesian(ScalarField3D self):
    cpdef VectorField3D _get_gradient_cartesian(ScalarField3D self):
        gg = np.gradient(
            self.values,
            *[
                self.node_intervals[iax]
                for iax in range(3) if not self.iax_isnull[iax]
            ],
            axis=[
                iax
                for iax in range(3)
                if not self.iax_isnull[iax]
            ]
        )
        gg = np.moveaxis(np.stack(gg), 0, -1)
        for iax in range(3):
            if self.iax_isnull[iax]:
                gg = np.insert(gg, iax, np.zeros(self.npts), axis=-1)
        grad = VectorField3D(coord_sys=self.coord_sys)
        grad.min_coords = self.min_coords
        grad.node_intervals = self.node_intervals
        grad.values = gg
        return (grad)


    cpdef VectorField3D _get_gradient_spherical(ScalarField3D self):
        grid       = self.nodes
        d0, d1, d2 = self.node_intervals
        n0, n1, n2 = self.npts

        if not self.iax_isnull[0]:
            # Second-order forward difference evaluated along the lower edge
            g0_lower = (
                (
                        self.values[2]
                    - 4*self.values[1]
                    + 3*self.values[0]
                ) / (2*d0)
            ).reshape(1, n1, n2)
            # Second order central difference evaluated in the interior
            g0_interior = (self.values[2:] - self.values[:-2]) / (2*d0)
            # Second order backward difference evaluated along the upper edge
            g0_upper = (
                (
                        self.values[-3]
                    - 4*self.values[-2]
                    + 3*self.values[-1]
                ) / (2*d0)
            ).reshape(1, n1, n2)
            g0 = np.concatenate([g0_lower, g0_interior, g0_upper], axis=0)
        else:
            g0 = np.zeros((n0, n1, n2))

        if not self.iax_isnull[1]:
            # Second-order forward difference evaluated along the lower edge
            g1_lower = (
                (
                        self.values[:,2]
                    - 4*self.values[:,1]
                    + 3*self.values[:,0]
                ) / (2*grid[:,0,:,0]*d1)
            ).reshape(n0, 1, n2)
            # Second order central difference evaluated in the interior
            g1_interior = (
                  self.values[:,2:]
                - self.values[:,:-2]
            ) / (2*grid[:,1:-1,:,0]*d1)
            # Second order backward difference evaluated along the upper edge
            g1_upper = (
                (
                        self.values[:,-3]
                    - 4*self.values[:,-2]
                    + 3*self.values[:,-1]
                ) / (2*grid[:,-1,:,0]*d1)
            ).reshape(n0, 1, n2)
            g1 = np.concatenate([g1_lower, g1_interior, g1_upper], axis=1)
        else:
            g1 = np.zeros((n0, n1, n2))

        if not self.iax_isnull[2]:
            # Second-order forward difference evaluated along the lower edge
            g2_lower = (
                  (
                        self.values[:,:,2]
                    - 4*self.values[:,:,1]
                    + 3*self.values[:,:,0]
                ) / (2*grid[:,:,0,0]*np.sin(grid[:,:,0,1])*d2)
            ).reshape(n0, n1, 1)

            # Second order central difference evaluated in the interior
            g2_interior = (
                  self.values[:,:,2:]
                - self.values[:,:,:-2]
            ) / (2*grid[:,:,1:-1,0]*np.sin(grid[:,:,1:-1,1])*d2)
            # Second order backward difference evaluated along the upper edge
            g2_upper = (
                (
                        self.values[:,:,-3]
                    - 4*self.values[:,:,-2]
                    + 3*self.values[:,:,-1]
                ) / (2*grid[:,:,-1,0]*np.sin(grid[:,:,-1,1])*d2)
            ).reshape(n0, n1, 1)
            g2 = np.concatenate([g2_lower, g2_interior, g2_upper], axis=2)
        else:
            g2 = np.zeros((n0, n1, n2))
        gg = np.moveaxis(np.stack([g0, g1, g2]), 0, -1)
        grad = VectorField3D(coord_sys=self.coord_sys)
        grad.min_coords = self.min_coords
        grad.node_intervals = self.node_intervals
        grad.values = gg
        return (grad)

cdef class VectorField3D(Field3D):

    def __init__(self, coord_sys="cartesian"):
        super(VectorField3D, self).__init__(coord_sys=coord_sys)


    @property
    def values(self):
        return (None if self._values is None else np.asarray(self._values))

    @values.setter
    def values(self, value):
        self._values = np.asarray(value)


    cpdef np.ndarray[constants.REAL_t, ndim=1] value(VectorField3D self, constants.REAL_t[:] point):
        '''
        Interpolate the contained field at *point*.
        :param point: Coordinates of the point to interpolate at.
        :type point: np.ndarray[_REAL_t, ndim=1]
        :return: Value of the field at *point*.
        :rtype: REAL_t
        :raises: OutOfBoundsError
        '''
        cdef constants.REAL_t[3]         ff
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
            if self.iax_isnull[iax]:
                ii[iax][0] = 0
                ii[iax][1] = 0
            else:
                ii[iax][0]  = <Py_ssize_t>idx[iax]
                ii[iax][1]  = <Py_ssize_t>(ii[iax][0]+1) % self.npts[iax]
            delta[iax] = idx[iax] % 1

        for iax in range(3):
            f000    = self.values[ii[0][0], ii[1][0], ii[2][0], iax]
            f100    = self.values[ii[0][1], ii[1][0], ii[2][0], iax]
            f110    = self.values[ii[0][1], ii[1][1], ii[2][0], iax]
            f101    = self.values[ii[0][1], ii[1][0], ii[2][1], iax]
            f111    = self.values[ii[0][1], ii[1][1], ii[2][1], iax]
            f010    = self.values[ii[0][0], ii[1][1], ii[2][0], iax]
            f011    = self.values[ii[0][0], ii[1][1], ii[2][1], iax]
            f001    = self.values[ii[0][0], ii[1][0], ii[2][1], iax]
            f00     = f000 + (f100 - f000) * delta[0]
            f10     = f010 + (f110 - f010) * delta[0]
            f01     = f001 + (f101 - f001) * delta[0]
            f11     = f011 + (f111 - f011) * delta[0]
            f0      = f00  + (f10  - f00)  * delta[1]
            f1      = f01  + (f11  - f01)  * delta[1]
            f       = f0   + (f1   - f0)   * delta[2]
            ff[iax] = f
        return (np.asarray(ff))
