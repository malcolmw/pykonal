cimport numpy as np

from libcpp cimport bool as bool_t

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
    cdef constants.REAL_t[:]     cy_arrivals_sorted
    cdef list                    cy_traveltimes_sorted

    cpdef bool_t add_arrivals(EQLocator self, dict arrivals)
    cpdef bool_t cleanup(EQLocator self)
    cpdef bool_t clear_arrivals(EQLocator self)
    cpdef bool_t compute_traveltime_lookup_table(EQLocator self, str station_id, str phase)
    cpdef bool_t compute_all_traveltime_lookup_tables(EQLocator self, str phase)
    cpdef bool_t load_traveltimes(EQLocator self)
    cpdef np.ndarray[constants.REAL_t, ndim=1] grid_search(EQLocator self)
    cpdef constants.REAL_t cost(EQLocator self, constants.REAL_t[:] hypocenter)
    cpdef np.ndarray[constants.REAL_t, ndim=1] locate(EQLocator self)
