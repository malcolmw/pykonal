cimport numpy as np

from libcpp cimport bool as bool_t

from . cimport constants
from . cimport fields
from . cimport heapq

cdef class EikonalSolver(object):
    cdef str                       cy_coord_sys
    cdef fields.ScalarField3D      cy_velocity
    cdef fields.ScalarField3D      cy_traveltime
    cdef heapq.Heap                cy_trial
    cdef constants.BOOL_t[:,:,:]   cy_known
    cdef constants.BOOL_t[:,:,:]   cy_unknown
    cdef constants.REAL_t[:,:,:,:] cy_norm
    cdef constants.UINT_t[3]       cy_is_periodic

    cpdef constants.BOOL_t solve(EikonalSolver self)
    cpdef np.ndarray[constants.REAL_t, ndim=2] trace_ray(
            EikonalSolver self,
            constants.REAL_t[:] end
    )
