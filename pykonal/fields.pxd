cimport numpy as np

from . cimport constants


cdef class Field3D(object):
    cdef str                     cy_coord_sys
    cdef constants.REAL_t[3]     cy_max_coords
    cdef constants.REAL_t[3]     cy_min_coords
    cdef constants.UINT_t[3]     cy_npts
    cdef constants.REAL_t[3]     cy_node_intervals
    cdef constants.BOOL_t[3]     cy_iax_isnull
    cdef constants.BOOL_t[3]     cy_iax_isperiodic

    cdef constants.BOOL_t _update_max_coords(Field3D self)
    cdef constants.BOOL_t _update_iax_isnull(Field3D self)
    cdef constants.BOOL_t _update_iax_isperiodic(Field3D self)


cdef class ScalarField3D(Field3D):
    cdef constants.REAL_t[:,:,:] cy_values

    cpdef np.ndarray[constants.REAL_t, ndim=1] resample(ScalarField3D self, constants.REAL_t[:,:] points, constants.REAL_t null=*)
    cpdef constants.REAL_t value(ScalarField3D self, constants.REAL_t[:] point, constants.REAL_t null=*)
    cpdef VectorField3D _gradient_of_cartesian(ScalarField3D self)
    cpdef VectorField3D _gradient_of_spherical(ScalarField3D self)


cdef class VectorField3D(Field3D):
    cdef constants.REAL_t[:,:,:,:] cy_values

    cpdef np.ndarray[constants.REAL_t, ndim=1] value(VectorField3D self, constants.REAL_t[:] point)


cpdef Field3D load(str path)
