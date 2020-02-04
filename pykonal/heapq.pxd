# distutils: language=c++

from libcpp.vector cimport vector as cpp_vector
from libcpp cimport bool as bool_t
from . cimport constants

cdef struct Index3D:
    Py_ssize_t i1, i2, i3

cdef class Heap(object):
    cdef cpp_vector[Index3D]     _keys
    cdef constants.REAL_t[:,:,:] _values
    cdef Py_ssize_t[:,:,:]       _heap_index

    cpdef tuple pop(Heap self)
    cpdef bool_t push(Heap self, Py_ssize_t i1, Py_ssize_t i2, Py_ssize_t i3)
    cpdef bool_t sift_down(Heap self, Py_ssize_t j_start, Py_ssize_t j)
    cpdef bool_t sift_up(Heap self, Py_ssize_t j_start)
