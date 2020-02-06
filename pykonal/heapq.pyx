"""
A module providing heap-sort functionality.

This module provides a single class (:class:`pykonal.heapq.Heap`) which
implements a binary min-heap structure whose elements are indices
(each having three components) sorted by an auxiliary value. Indices can
be pushed onto and popped from the Heap and the auxiliary sort values
can be updated on the fly, although care must be taken to resort the
Heap so as to maintain the heap invariant when updating the underlying
sort values.
"""

import numpy as np

cimport numpy as np

from libcpp cimport bool as bool_t

from . import constants


cdef struct Index3D:
    Py_ssize_t i1, i2, i3


cdef class Heap(object):
    """
    Binary Min-Heap structure for storing indices sorted by auxiliary
    values.
    """
    def __init__(self, values):
        self._values     = values
        self._heap_index = np.full(values.shape, fill_value=-1)


    @property
    def heap_index(self):
        """
        [:class:`numpy.ndarray`\ (shape=(N0,N1,N2), dtype=numpy.int)]
        Array of indices indicating the heap position of each node. 
        Index -1 indicates that a node is not on the heap.
        """
        return (np.asarray(self._heap_index, dtype=constants.DTYPE_INT))

    @property
    def keys(self):
        """
        [:class:`list`\ [(int, int, int),...]] Sorted list of node
        indices currently on the heap.
        """
        cdef Index3D idx
        output = []
        for i in range(self._keys.size()):
            idx = self._keys[i]
            output.append((idx.i1, idx.i2, idx.i3))
        return (output)


    @property
    def size(self):
        """
        [:class:`int`] Number of node indices on the heap.
        """
        return (self._keys.size())


    @property
    def values(self):
        """
        [:class:`numpy.ndarray`\ (shape=(N0,N1,N2), dtype=numpy.float)]
        Auxiliary values to sort by.
        """
        return (np.asarray(self._values))

    @values.setter
    def values(self, values):
        self._values = values


    cpdef (Py_ssize_t, Py_ssize_t, Py_ssize_t) pop(Heap self):
        """
        pop(self)

        Pop the index of the item with the smallest sort value from the
        heap, maintaining the heap invariant.

        :return: Index of node on the heap with smallest sort value.
        :rtype: tuple(int, int, int)
        """
        cdef Index3D last, idx_return

        last = self._keys.back()
        self._keys.pop_back()
        self._heap_index[last.i1, last.i2, last.i3] = -1
        if self._keys.size() > 0:
            idx_return = self._keys[0]
            self._heap_index[idx_return.i1, idx_return.i2, idx_return.i3] = -1
            self._keys[0] = last
            self._heap_index[last.i1, last.i2, last.i3] = 0
            self.sift_up(0)
            return ((idx_return.i1, idx_return.i2, idx_return.i3))
        return ((last.i1, last.i2, last.i3))

    cpdef bool_t push(Heap self, Py_ssize_t i1, Py_ssize_t i2, Py_ssize_t i3):
        """
        push(self, i1, i2, i3)

        Push index onto the heap, maintaining the heap invariant.

        :param i1: First component of index.
        :type i1: int
        :param i2: Second component of index.
        :type i2: int
        :param i3: Third component of index.
        :type i3: int

        :return: True upon successful completion.
        :rtype: bool

        .. todo:: Check that index is in range.
        """
        cdef Index3D idx
        idx.i1, idx.i2, idx.i3 = i1, i2, i3
        self._keys.push_back(idx)
        self._heap_index[idx.i1, idx.i2, idx.i3] = self._keys.size()-1
        self.sift_down(0, self._keys.size()-1)
        return (True)


    cpdef bool_t sift_down(Heap self, Py_ssize_t j_start, Py_ssize_t j):
        """
        sift_down(self, j_start, j)

        Sift the heap element at *j* down the heap (towards the root),
        without going past *j_start*, until finding a place that it
        fits.

        :param j_start: Heap index creating a barrier beyond which heap
                        element *j* cannot be sifted.
        :type j_start: int
        :param j: Heap index of element to sift towards root.
        :type j: int
        :return: Returns True upon successful execution.
        :rtype: bool
        """
        cdef Py_ssize_t j_parent
        cdef Index3D    idx_new, idx_parent

        idx_new = self._keys[j]
        # Follow the path to the root, moving parents down until finding a place
        # new item fits.
        while j > j_start:
            j_parent = (j - 1) >> 1
            idx_parent = self._keys[j_parent]
            if self._values[idx_new.i1, idx_new.i2, idx_new.i3] < self._values[idx_parent.i1, idx_parent.i2, idx_parent.i3]:
                self._keys[j] = idx_parent
                self._heap_index[idx_parent.i1, idx_parent.i2, idx_parent.i3] = j
                j = j_parent
                continue
            break
        self._keys[j] = idx_new
        self._heap_index[idx_new.i1, idx_new.i2, idx_new.i3] = j
        return (True)


    cpdef bool_t sift_up(Heap self, Py_ssize_t j_start):
        """
        sift_up(self, j_start)

        Sift the heap element at *j_start* up the heap (away from the
        root), until finding a place that it fits.

        :param j_start: Heap index of element to sift away from root.
        :type j_start: int
        :return: Returns True upon successful execution.
        :rtype: bool
        """
        cdef Py_ssize_t j, j_child, j_end, j_right
        cdef Index3D idx_child, idx_right, idx_new

        j_end = self._keys.size()
        j = j_start
        idx_new = self._keys[j_start]
        # Bubble up the smaller child until hitting a leaf.
        j_child = 2 * j_start + 1 # leftmost child position
        while j_child < j_end:
            # Set childpos to index of smaller child.
            j_right = j_child + 1
            idx_child, idx_right = self._keys[j_child], self._keys[j_right]
            if j_right < j_end and not self._values[idx_child.i1, idx_child.i2, idx_child.i3] < self._values[idx_right.i1, idx_right.i2, idx_right.i3]:
                j_child = j_right
            # Move the smaller child up.
            self._keys[j] = self._keys[j_child]
            self._heap_index[self._keys[j_child].i1, self._keys[j_child].i2, self._keys[j_child].i3] = j
            j = j_child
            j_child = 2 * j + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self._keys[j] = idx_new
        self._heap_index[idx_new.i1, idx_new.i2, idx_new.i3] = j
        self.sift_down(j_start, j)
        return (True)
