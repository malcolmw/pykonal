cimport numpy as np

from . cimport constants
from . cimport fields


cdef class EQLocator(object):
    cdef str                     cy_coord_sys
    cdef dict                    cy_stations
    cdef str                     cy_tt_dir
    cdef object                  cy_tempdir_obj
    cdef fields.ScalarField3D    cy_grid
    cdef fields.ScalarField3D    cy_pwave_velocity
    cdef fields.ScalarField3D    cy_swave_velocity
    cdef dict                    cy_arrivals
    cdef dict                    cy_traveltimes
    cdef dict                    cy_residual_rvs
    cdef constants.REAL_t[:]     cy_arrivals_sorted
    cdef list                    cy_traveltimes_sorted

    cpdef constants.BOOL_t add_arrivals(EQLocator self, dict arrivals)
    cpdef constants.BOOL_t add_residual_rvs(EQLocator self, dict residua_rvs)
    cpdef constants.BOOL_t cleanup(EQLocator self)
    cpdef constants.BOOL_t clear_arrivals(EQLocator self)
    cpdef constants.BOOL_t clear_residual_rvs(EQLocator self)
    cpdef constants.BOOL_t compute_traveltime_lookup_table(
        EQLocator self,
        str station_id,
        str phase
    )
    cpdef constants.BOOL_t compute_all_traveltime_lookup_tables(
        EQLocator self,
        str phase
    )
    cpdef constants.BOOL_t load_traveltimes(EQLocator self)
    cpdef constants.REAL_t log_likelihood(
        EQLocator self,
        constants.REAL_t[:] model
    )
    cpdef np.ndarray[constants.REAL_t, ndim=1] grid_search(EQLocator self)
    cpdef constants.REAL_t rms(EQLocator self, constants.REAL_t[:] hypocenter)
    cpdef np.ndarray[constants.REAL_t, ndim=1] locate(
        EQLocator self,
        constants.REAL_t dlat=*,
        constants.REAL_t dlon=*,
        constants.REAL_t dz=*,
        constants.REAL_t dt=*
    )
