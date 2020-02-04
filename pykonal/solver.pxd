cimport numpy as np

from libcpp cimport bool as bool_t

from . cimport constants
from . cimport fields
from . cimport heapq

cdef class EikonalSolver(object):
    cdef str                       _coord_sys
    cdef fields.ScalarField3D      _velocity
    cdef fields.ScalarField3D      _traveltime
    cdef heapq.Heap                _trial
    cdef constants.BOOL_t[:,:,:]   _known
    cdef constants.BOOL_t[:,:,:]   _unknown
    cdef constants.REAL_t[:,:,:,:] _norm
    cdef constants.UINT_t[3]       _is_periodic
    cpdef bool_t solve(EikonalSolver self)
    cpdef np.ndarray[constants.REAL_t, ndim=2] trace_ray(
            EikonalSolver self,
            constants.REAL_t[:] end
    )
