cimport numpy as np

from . cimport constants
from . cimport fields


cdef class EQLocator(object):
    cdef str                     cy_coord_sys
    cdef dict                    cy_stations
    cdef object                  cy_traveltime_inventory
    cdef fields.ScalarField3D    cy_grid
    cdef fields.ScalarField3D    cy_pwave_velocity
    cdef fields.ScalarField3D    cy_swave_velocity
    cdef dict                    cy_arrivals
    cdef dict                    cy_traveltimes
    cdef dict                    cy_residual_rvs

    cpdef constants.BOOL_t add_arrivals(EQLocator self, dict arrivals)
    cpdef constants.BOOL_t add_residual_rvs(EQLocator self, dict residua_rvs)
    cpdef constants.BOOL_t clear_arrivals(EQLocator self)
    cpdef constants.BOOL_t clear_residual_rvs(EQLocator self)
    cpdef constants.BOOL_t read_traveltimes(
        EQLocator self,
        constants.REAL_t[:] min_coords=*,
        constants.REAL_t[:] max_coords=*
    )
    #cpdef constants.REAL_t log_likelihood(
    #    EQLocator self,
    #    constants.REAL_t[:] model
    #)
    #cpdef np.ndarray[constants.REAL_t, ndim=1] grid_search(EQLocator self)
    cpdef constants.REAL_t rms(EQLocator self, constants.REAL_t[:] hypocenter)
    cpdef np.ndarray[constants.REAL_t, ndim=1] locate(
        EQLocator self,
        np.ndarray[constants.REAL_t, ndim=1] initial,
        np.ndarray[constants.REAL_t, ndim=1] delta
    )
