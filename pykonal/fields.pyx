# distutils: language=c++
"""
This module provides three classes (:class:`Field3D <pykonal.fields.Field3D>`,
:class:`ScalarField3D <pykonal.fields.ScalarField3D>`, and
:class:`VectorField3D <pykonal.fields.VectorField3D>`). 
:class:`Field3D <pykonal.fields.Field3D>` is a base class intended to
be used only as an abstraction. :class:`ScalarField3D <pykonal.fields.ScalarField3D>`
and :class:`VectorField3D <pykonal.fields.VectorField3D>` both inherit
from :class:`Field3D <pykonal.fields.Field3D>` and add important
functionality that make the classes useful. They are primarily intended
for storing and interpolating data, and, in the case of scalar fields,
computing the gradient.

This module also provides an I/O function
(:func:`load <pykonal.fields.load>`) to load data that was saved using
the :func:`savez <pykonal.fields.Field3D.savez>` method of the above
classes.

.. autofunction:: pykonal.fields.load(path)
"""

# Third-party imports
import numpy as np

# Local imports
from . import constants
from . import transformations


# Third-party Cython imports
cimport numpy as np

# Local Cython imports
from . cimport constants

cdef class Field3D(object):
    """
    Base class for representing generic 3D fields.
    """

    def __init__(self, coord_sys="cartesian"):
        self.cy_coord_sys = coord_sys

    @property
    def coord_sys(self):
        """
        [*Read only*, :class:`str`] Coordinate system of grid on which
        field data is represented {"Cartesian", "spherical"}.
        """
        return (self.cy_coord_sys)

    @property
    def iax_isnull(self):
        """
        [*Read only*, :class:`numpy.ndarray`\ (shape=(3,), dtype=numpy.bool)]
        Array of booleans indicating whether each axis is null. The
        axis with only one layer of nodes in 2D problems will be null.
        """
        arr = np.array(
            [self.cy_iax_isnull[iax] for iax in range(3)],
            dtype=constants.DTYPE_BOOL
        )
        return (arr)

    @property
    def iax_isperiodic(self):
        """
        [*Read only*, :class:`numpy.ndarray`\ (shape=(3,), dtype=numpy.bool)]
        Array of booleans indicating whether each axis is periodic. In
        practice, only the azimuthal (:math:`\phi`) axis in spherical
        coordinates will ever be periodic.
        """
        arr = np.array(
            [self.cy_iax_isperiodic[iax] for iax in range(3)],
            dtype=constants.DTYPE_BOOL
        )
        return (arr)

    @property
    def node_intervals(self):
        """
        [*Read/Write*, :class:`numpy.ndarray`\ (shape=(3,), dtype=numpy.float)]
        Array of node intervals along each axis. This attribute must be
        initialized by the user.
        """
        return (np.asarray(self.cy_node_intervals))

    @node_intervals.setter
    def node_intervals(self, value):
        value = np.asarray(value, dtype=constants.DTYPE_REAL)
        if np.any(value) <= 0:
            raise (ValueError("All node intervals must be > 0"))
        self.cy_node_intervals = value
        self._update_max_coords()
        self._update_iax_isperiodic()


    @property
    def npts(self):
        """
        [*Read/Write*, :class:`numpy.ndarray`\ (shape=(3,), dtype=numpy.int)]
        Array specifying the number of nodes along each axis. This
        attribute must be initialized by the user.
        """
        return (np.asarray(self.cy_npts))

    @npts.setter
    def npts(self, value):
        self.cy_npts = np.asarray(value, dtype=constants.DTYPE_UINT)
        self._update_max_coords()
        self._update_iax_isnull()
        self._update_iax_isperiodic()

    @property
    def min_coords(self):
        """
        [*Read/Write*, :class:`numpy.ndarray`\ (shape=(3,), dtype=numpy.float)]
        Array specifying the lower bound of each axis. This attribute
        must be initialized by the user.
        """
        return (np.asarray(self.cy_min_coords))

    @min_coords.setter
    def min_coords(self, value):
        if self.coord_sys == "spherical" and value[0] == 0:
            raise (ValueError("min_coords[0] must be > 0 for spherical coordinates."))
        self.cy_min_coords = np.asarray(value, dtype=constants.DTYPE_REAL)
        self._update_max_coords()
        self._update_iax_isperiodic()


    @property
    def max_coords(self):
        """
        [*Read only*, :class:`numpy.ndarray`\ (shape=(3,), dtype=numpy.float)]
        Array specifying the upper bound of each axis.
        """
        return (np.asarray(self.cy_max_coords))


    @property
    def nodes(self):
        """
        [*Read only*, :class:`numpy.ndarray`\ (shape=(N0,N1,N2,3), dtype=numpy.float)]
        Array specifying the grid-node coordinates.
        """
        nodes = [
            np.linspace(
                self.min_coords[idx],
                self.max_coords[idx],
                self.npts[idx],
                dtype=constants.DTYPE_REAL
            )
            for idx in range(3)
        ]
        nodes = np.meshgrid(*nodes, indexing="ij")
        nodes = np.stack(nodes)
        nodes = np.moveaxis(nodes, 0, -1)
        return (nodes)


    cdef constants.BOOL_t _update_iax_isperiodic(Field3D self):
        if self.cy_coord_sys == "spherical":
            self.cy_iax_isperiodic[2] = np.isclose(
                self.cy_max_coords[2]+self.cy_node_intervals[2]-self.cy_min_coords[2],
                2 * np.pi
            )
        return (True)


    cdef constants.BOOL_t _update_iax_isnull(Field3D self):
        cdef Py_ssize_t       iax

        for iax in range(3):
            self.cy_iax_isnull[iax] = True if self.cy_npts[iax] == 1 else False
        return (True)


    cdef constants.BOOL_t _update_max_coords(Field3D self):
        cdef Py_ssize_t       iax
        cdef constants.REAL_t dx

        for iax in range(3):
            dx = self.cy_node_intervals[iax] * <constants.REAL_t>(self.cy_npts[iax] - 1)
            self.cy_max_coords[iax] = self.cy_min_coords[iax] + dx
        return (True)


    def savez(self, path):
        """
        savez(self, path)

        Save the field to disk using numpy.savez.

        :param path: Path to output file.
        :type path: str

        :return: Returns True upon successful execution.
        :rtype: bool
        """
        np.savez_compressed(
            path,
            min_coords=self.min_coords,
            node_intervals=self.node_intervals,
            npts=self.npts,
            values=self.values,
            coord_sys=[self.coord_sys]
        )
        return (True)

    def transform_coordinates(self, coord_sys, origin):
        """
        transform_coordinates(self, coord_sys, origin)

        Transform node coordinates to a new frame of reference.

        :param coord_sys: Coordinate system to transform to
                          ("*spherical*", or "*Cartesian*")
        :type coord_sys: str
        :param origin: Coordinates of the origin of the new frame with
                       respect to the old frame of reference.
        :type origin: tuple(float, float, float)
        :return: Node coordinates in new frame of reference.
        :rtype: numpy.ndarray(shape=(N0,N1,N2,3), dtype=numpy.float)
        """
        if self.coord_sys == "spherical" and coord_sys.lower() == "spherical":
            return (transformations.sph2sph(self.nodes, origin))
        elif self.coord_sys == "cartesian" and coord_sys.lower() == "spherical":
            return (transformations.xyz2sph(self.nodes, origin))
        elif self.coord_sys == "spherical" and coord_sys.lower() == "cartesian":
            return (transformations.sph2xyz(self.nodes, origin))
        elif self.coord_sys == "cartesian" and coord_sys.lower() == "cartesian":
            return (transformations.xyz2xyz(self.nodes, origin))
        else:
            raise (NotImplementedError())



cdef class ScalarField3D(Field3D):
    """
    Class for representing 3D scalar fields.
    """
    def __init__(self, coord_sys="cartesian"):
        super(ScalarField3D, self).__init__(coord_sys=coord_sys)

    @property
    def gradient(self):
        """
        [*Read only*, :class:`numpy.ndarray`\ (shape=(N0,N1,N2,3), dtype=numpy.float)]
        Gradient of the field.
        """
        if self.coord_sys == "cartesian":
            return (self._gradient_of_cartesian())
        elif self.coord_sys == "spherical":
            return (self._gradient_of_spherical())


    @property
    def values(self):
        """
        [*Read/Write*, :class:`numpy.ndarray`\ (shape=(N0,N1,N2), dtype=numpy.float)]
        Value of the field at each grid node.
        """
        try:
            return (np.asarray(self.cy_values))
        except AttributeError:
            self.cy_values = np.full(self.npts, fill_value=np.nan)
        return (np.asarray(self.cy_values))

    @values.setter
    def values(self, value):
        values = np.asarray(value, dtype=constants.DTYPE_REAL)
        if not np.all(values.shape == self.npts):
            raise (ValueError("Shape of values does not match npts attribute."))
        self.cy_values = values


    cpdef np.ndarray[constants.REAL_t, ndim=1] resample(ScalarField3D self, constants.REAL_t[:,:] points, constants.REAL_t null=np.nan):
        """
        resample(self, points, null=numpy.nan)

        Resample the field at an arbitrary set of points using
        trilinear interpolation.

        :param points: Points at which to resample the field.
        :type points: numpy.ndarray(shape=(N,3), dtype=numpy.float)
        :param null: Default (null) value to return for points lying
                     outside the interpolation domain.
        :type null: float
        :return: Resampled field values.
        :rtype: numpy.ndarray(shape=(N,), dtype=numpy.float)
        """
        cdef Py_ssize_t                           idx
        cdef np.ndarray[constants.REAL_t, ndim=1] resampled # Using a MemoryViewBuffer might make this faster.

        resampled = np.empty(points.shape[0], dtype=constants.DTYPE_REAL)

        for idx in range(points.shape[0]):
            resampled[idx] = self.value(points[idx], null=null)

        return (resampled)


    cpdef constants.REAL_t value(ScalarField3D self, constants.REAL_t[:] point, constants.REAL_t null=np.nan):
        """
        value(self, point, null=numpy.nan)

        Interpolate the field at *point* using trilinear interpolation.

        :param point: Coordinates of the point at which to interpolate
                      the field.
        :type point: numpy.ndarray(shape=(3,), dtype=numpy.float)
        :param null: Default (null) value to return if point lies
                     outside the interpolation domain.
        :type null: float
        :return: Value of the field at *point*.
        :rtype: float
        """
        cdef constants.REAL_t[3]         delta, idx
        cdef constants.REAL_t            f000, f100, f110, f101, f111, f010, f011, f001
        cdef constants.REAL_t            f00, f10, f01, f11
        cdef constants.REAL_t            f0, f1
        cdef constants.REAL_t            f
        cdef Py_ssize_t[3][2]            ii
        cdef Py_ssize_t                  i1, i2, i3, iax, di1, di2, di3

        for iax in range(3):
            if (
                (
                    point[iax] < self.cy_min_coords[iax]
                    or point[iax] > self.cy_max_coords[iax]
                )
                and not self.cy_iax_isperiodic[iax]
                and not self.cy_iax_isnull[iax]
            ):
                return (null)
            idx[iax]   = (point[iax] - self.cy_min_coords[iax]) / self.cy_node_intervals[iax]
            if self.cy_iax_isnull[iax]:
                ii[iax][0] = 0
                ii[iax][1] = 0
            else:
                ii[iax][0]  = <Py_ssize_t>idx[iax]
                ii[iax][1]  = <Py_ssize_t>(ii[iax][0]+1) % self.npts[iax]
            delta[iax] = idx[iax] % 1
        f000    = self.cy_values[ii[0][0], ii[1][0], ii[2][0]]
        f100    = self.cy_values[ii[0][1], ii[1][0], ii[2][0]]
        f110    = self.cy_values[ii[0][1], ii[1][1], ii[2][0]]
        f101    = self.cy_values[ii[0][1], ii[1][0], ii[2][1]]
        f111    = self.cy_values[ii[0][1], ii[1][1], ii[2][1]]
        f010    = self.cy_values[ii[0][0], ii[1][1], ii[2][0]]
        f011    = self.cy_values[ii[0][0], ii[1][1], ii[2][1]]
        f001    = self.cy_values[ii[0][0], ii[1][0], ii[2][1]]
        f00     = f000 + (f100 - f000) * delta[0]
        f10     = f010 + (f110 - f010) * delta[0]
        f01     = f001 + (f101 - f001) * delta[0]
        f11     = f011 + (f111 - f011) * delta[0]
        f0      = f00  + (f10  - f00)  * delta[1]
        f1      = f01  + (f11  - f01)  * delta[1]
        f       = f0   + (f1   - f0)   * delta[2]
        return (f)


    cpdef VectorField3D _gradient_of_cartesian(ScalarField3D self):
        """
        The gradient of a field represented on a Cartesian grid.
        """
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
        grad.npts = self.npts
        grad.values = gg
        return (grad)


    cpdef VectorField3D _gradient_of_spherical(ScalarField3D self):
        """
        The gradient of a field represented on a spherical grid.
        """
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
        grad.npts = self.npts
        grad.values = gg
        return (grad)

cdef class VectorField3D(Field3D):
    """
    Class for representing 3D vector fields.
    """

    def __init__(self, coord_sys="cartesian"):
        super(VectorField3D, self).__init__(coord_sys=coord_sys)


    @property
    def values(self):
        """
        [*Read/Write*, :class:`numpy.ndarray`\ (shape=(N0,N1,N2,3), dtype=numpy.float)]
        Value of the field at each grid node.
        """
        try:
            return (np.asarray(self.cy_values))
        except AttributeError:
            self.cy_values = np.full(self.npts, fill_value=np.nan)
        return (np.asarray(self.cy_values))

    @values.setter
    def values(self, value):
        values = np.asarray(value)
        if not np.all(values.shape[:3] == self.npts):
            raise (ValueError("Shape of values does not match npts attribute."))
        self.cy_values = np.asarray(value)


    cpdef np.ndarray[constants.REAL_t, ndim=1] value(VectorField3D self, constants.REAL_t[:] point):
        """
        value(self, point)

        Interpolate the field at *point* using trilinear interpolation.

        :param point: Coordinates of the point at which to interpolate
                      the field.
        :type point: numpy.ndarray(shape=(3,), dtype=numpy.float)
        :return: Value of the field at *point*.
        :rtype: numpy.ndarray(shape=(3,), dtype=numpy.float)
        """
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
            #    and not self.is_periodic[iax]
            #    and not self._iax_isnull[iax]
            #):
            #    raise(
            #        OutOfBoundsError(
            #            f"Point outside of interpolation domain requested: ({point[0]}, {point[1]}, {point[2]})"
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
            f000    = self.cy_values[ii[0][0], ii[1][0], ii[2][0], iax]
            f100    = self.cy_values[ii[0][1], ii[1][0], ii[2][0], iax]
            f110    = self.cy_values[ii[0][1], ii[1][1], ii[2][0], iax]
            f101    = self.cy_values[ii[0][1], ii[1][0], ii[2][1], iax]
            f111    = self.cy_values[ii[0][1], ii[1][1], ii[2][1], iax]
            f010    = self.cy_values[ii[0][0], ii[1][1], ii[2][0], iax]
            f011    = self.cy_values[ii[0][0], ii[1][1], ii[2][1], iax]
            f001    = self.cy_values[ii[0][0], ii[1][0], ii[2][1], iax]
            f00     = f000 + (f100 - f000) * delta[0]
            f10     = f010 + (f110 - f010) * delta[0]
            f01     = f001 + (f101 - f001) * delta[0]
            f11     = f011 + (f111 - f011) * delta[0]
            f0      = f00  + (f10  - f00)  * delta[1]
            f1      = f01  + (f11  - f01)  * delta[1]
            f       = f0   + (f1   - f0)   * delta[2]
            ff[iax] = f
        return (np.asarray(ff))

cpdef Field3D load(str path):
    """
    Load field data from disk.

    :param path: Path to input file.
    :type path: str
    :return: A Field3D-derivative class initialized with data in *path*.
    :rtype: ScalarField3D or VectorField3D
    """
    with np.load(path) as npz:
        coord_sys = str(npz["coord_sys"][0])
        if len(npz["values"].shape) == 4:
            field = VectorField3D(coord_sys=coord_sys)
        else:
            field = ScalarField3D(coord_sys=coord_sys)
        field.min_coords = npz["min_coords"]
        field.node_intervals = npz["node_intervals"]
        field.npts = npz["npts"]
        field.values = npz["values"]
    return (field)
