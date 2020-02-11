# distutils: language=c++

from libcpp.vector cimport vector as cpp_vector
from . cimport constants

cdef struct Index3D:
    Py_ssize_t i1, i2, i3

cdef class Heap(object):
    cdef cpp_vector[Index3D]     cy_keys
    cdef constants.REAL_t[:,:,:] cy_values
    cdef Py_ssize_t[:,:,:]       cy_heap_index

    cpdef (Py_ssize_t, Py_ssize_t, Py_ssize_t) pop(Heap self)
    cpdef constants.BOOL_t push(Heap self, Py_ssize_t i1, Py_ssize_t i2, Py_ssize_t i3)
    cpdef constants.BOOL_t sift_down(Heap self, Py_ssize_t j_start, Py_ssize_t j)
    cpdef constants.BOOL_t sift_up(Heap self, Py_ssize_t j_start)
