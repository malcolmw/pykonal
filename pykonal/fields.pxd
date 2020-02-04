cimport numpy as np

from . cimport constants

cdef class Field3D(object):
    cdef str                     _coord_sys
    cdef constants.REAL_t[3]     _min_coords
    cdef constants.UINT_t[3]     _npts
    cdef constants.REAL_t[3]     _node_intervals

cdef class ScalarField3D(Field3D):
    cdef constants.REAL_t[:,:,:] _values

    cpdef np.ndarray[constants.REAL_t, ndim=1] resample(ScalarField3D self, constants.REAL_t[:,:] points, constants.REAL_t null=*)
    cpdef constants.REAL_t value(ScalarField3D self, constants.REAL_t[:] point, constants.REAL_t null=*)
    cpdef VectorField3D _gradient_of_cartesian(ScalarField3D self)
    cpdef VectorField3D _gradient_of_spherical(ScalarField3D self)


cdef class VectorField3D(Field3D):
    cdef constants.REAL_t[:,:,:,:] _values

    cpdef np.ndarray[constants.REAL_t, ndim=1] value(VectorField3D self, constants.REAL_t[:] point)
