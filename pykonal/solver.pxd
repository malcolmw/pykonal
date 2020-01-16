cimport numpy as np

from . cimport constants
from . cimport field
from . cimport heapq

cdef class EikonalSolver(object):
    cdef str                       _coord_sys
    cdef field.ScalarField3D       _velocity
    cdef field.ScalarField3D       _traveltime
    cdef heapq.Heap                _close
    cdef constants.BOOL_t[:,:,:]   _is_alive
    cdef constants.BOOL_t[:,:,:]   _is_far
    cdef constants.REAL_t[:,:,:,:] _norm
    cdef constants.UINT_t[3]       _is_periodic
    cpdef void solve(EikonalSolver self)
    cpdef np.ndarray[constants.REAL_t, ndim=2] trace_ray(
            EikonalSolver self,
            constants.REAL_t[:] end
    )
