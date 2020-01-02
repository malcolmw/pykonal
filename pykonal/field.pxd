from . cimport constants

cdef class Field3D(object):
    cdef str                     _coord_sys
    cdef constants.REAL_t[3]     _min_coords
    cdef constants.UINT_t[3]     _npts
    cdef constants.REAL_t[3]     _node_intervals
    cdef constants.REAL_t[:,:,:] _values

    cpdef constants.REAL_t value(self, constants.REAL_t[:] point) except? -999999999999.
