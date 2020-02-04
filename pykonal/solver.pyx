# Cython compiler directives.
# distutils: language=c++

"""
The core module of PyKonal for solving the Eikonal equation.

This module provides a single class (:class:`pykonal.solver.EikonalSolver`),
which gets imported into the root-level namespace and can thus be
instantiated as below:

.. code-block:: python

   import pykonal
   solver = pykonal.EikonalSolver(coord_sys="cartesian")
"""

# Python thrid-party imports.
import numpy as np

# Local imports.
from . import constants

# Cython built-in imports.
from libcpp cimport bool as bool_t
from libc.math cimport sqrt, sin
from libcpp.vector cimport vector as cpp_vector
from libc.stdlib   cimport malloc, free

# Cython third-party imports.
cimport numpy as np

# Cython local imports.
from . cimport constants
from . cimport fields
from . cimport heapq

cdef class EikonalSolver(object):
    """
    The core class of PyKonal for solving the Eikonal equation.

    .. code-block:: python

       import numpy as np
       import pykonal

       # Instantiate EikonalSolver object.
       solver = pykonal.solver.EikonalSolver(coord_sys="cartesian")
       # Initialize EikonalSolver object's velocity attribute.
       solver.velocity.min_coords = 0, 0, 0
       solver.velocity.node_intervals = 1, 1, 1
       solver.velocity.npts = 16, 16, 16
       solver.velocity.values = np.ones(solver.velocity.npts)
       # Initialize the traveltime field with a source at node with
       # index (0, 0, 0).
       src_idx = (0, 0, 0)
       # Remove source node from *Unknown*
       solver.unknown[src_idx] = False
       # Add source node to *Trial*.
       solver.trial.push(*src_idx)
       # Solve the system.
       solver.solve()
       # Extract the traveltime values.
       tt = solver.traveltime.values
    """

    def __init__(self, coord_sys="cartesian"):
        self._coord_sys = coord_sys
        self._velocity = fields.ScalarField3D(coord_sys=self.coord_sys)


    @property
    def trial(self):
        """
        [*Read/Write*, :class:`pykonal.heapq.Heap`] Heap of node
        indices in *Trial*.
        """
        if self._trial is None:
            self._trial = heapq.Heap(self.traveltime.values)
        return (self._trial)

    @property
    def coord_sys(self):
        """
        [*Read only*, :class:`str`] Coordinate system of solver
        {"Cartesian", "spherical"}.
        """
        return (self._coord_sys)


    @property
    def known(self):
        """
        [*Read/Write*, :class:`numpy.ndarray`\ (shape=(N0,N1,N2), dtype=numpy.bool)] 3D array of booleans
        indicating whether nodes are in *Known*.
        """
        try:
            return (np.asarray(self._known))
        except AttributeError:
            self._known = np.zeros(self.tt.npts, dtype=constants.DTYPE_BOOL)
        return (np.asarray(self._known))

    @property
    def unknown(self):
        """
        [*Read/Write*, :class:`numpy.ndarray`\ (shape=(N0,N1,N2), dtype=numpy.bool)] 3D array of booleans
        indicating whether nodes are in *Unknown*.
        """
        try:
            return (np.asarray(self._unknown))
        except AttributeError:
            self._unknown = np.ones(self.tt.npts, dtype=constants.DTYPE_BOOL)
        return (np.asarray(self._unknown))



    @property
    def norm(self):
        """
        [*Read-only*, :class:`numpy.ndarray`\ (shape=(N0,N1,N2,3), dtype=numpy.float)] 4D array of scaling
        factors for gradient operator.
        """
        try:
            return (np.asarray(self._norm))
        except AttributeError:
            norm = np.tile(
                self.traveltime.node_intervals,
                np.append(self.traveltime.npts, 1)
            )
            if self.coord_sys == 'spherical':
                norm[..., 1] *= self.traveltime.nodes[..., 0]
                norm[..., 2] *= self.traveltime.nodes[..., 0]
                norm[..., 2] *= np.sin(self.traveltime.nodes[..., 1])
            self._norm = norm
        return (np.asarray(self._norm))

    @property
    def step_size(self):
        """
        [*Read only*, :class:`float`] Step size used for ray tracing.
        """
        return (self.norm[~np.isclose(self.norm, 0)].min() / 4)


    @property
    def traveltime(self):
        """
        [*Read/Write*, :class:`pykonal.fields.ScalarField3D`] 3D array
        of traveltime values.
        """
        if self._traveltime is None:
            self._traveltime = fields.ScalarField3D(coord_sys=self.coord_sys)
            self._traveltime.min_coords = self.velocity.min_coords
            self._traveltime.node_intervals = self.velocity.node_intervals
            self._traveltime.npts = self.velocity.npts
            self._traveltime.values = np.full_like(self.velocity.values, fill_value=np.inf)
        return (self._traveltime)

    @property
    def tt(self):
        """
        [*Read/Write*, :class:`pykonal.fields.ScalarField3D`] Alias for
        self.traveltime.
        """
        return (self.traveltime)


    @property
    def velocity(self):
        """
        [*Read/Write*, :class:`pykonal.fields.ScalarField3D`] 3D array
        of velocity values.
        """
        return (self._velocity)

    @property
    def vv(self):
        """
        [*Read/Write*, :class:`pykonal.fields.ScalarField3D`] Alias for
        self.velocity.
        """
        return (self.velocity)


    cpdef bool_t solve(EikonalSolver self):
        """
        solve(self)

        Solve the Eikonal equation using the FMM.

        :return: Returns True upon successful execution.
        :rtype:  bool
        """
        cdef Py_ssize_t                i, iax, idrxn, active_i1, active_i2, active_i3, iheap
        cdef Py_ssize_t[6][3]          nbrs
        cdef Py_ssize_t[3]             active_idx, nbr, switch, max_idx
        cdef Py_ssize_t[2]             drxns = [-1, 1]
        cdef int                       count_a = 0
        cdef int                       count_b = 0
        cdef int                       inbr
        cdef int[2]                    order
        cdef constants.REAL_t          a, b, c, bfd, ffd, new
        cdef constants.REAL_t[2]       fdu
        cdef constants.REAL_t[3]       aa, bb, cc
        cdef constants.REAL_t[:,:,:]   tt, vv
        cdef constants.REAL_t[:,:,:,:] norm
        cdef constants.BOOL_t[:]       is_periodic,
        cdef constants.BOOL_t[:,:,:]   known, unknown
        cdef heapq.Heap                trial

        max_idx = self.traveltime.npts
        is_periodic = self.traveltime.is_periodic
        tt = self.traveltime.values
        vv = self.velocity.values
        norm = self.norm
        known = self.known
        unknown = self.unknown
        trial = self.trial


        while trial.size > 0:
            # Let Active be the point in Trial with the smallest value of u
            active_i1, active_i2, active_i3 = trial.pop()
            active_idx = [active_i1, active_i2, active_i3]
            known[active_i1, active_i2, active_i3] = True

            # Determine the indices of neighbouring nodes.
            inbr = 0
            for iax in range(3):
                switch = [0, 0, 0]
                for idrxn in range(2):
                    switch[iax] = drxns[idrxn]
                    for jax in range(3):
                        if is_periodic[jax]:
                            nbrs[inbr][jax] = (
                                  active_idx[jax]
                                + switch[jax]
                                + max_idx[jax]
                            )\
                            % max_idx[jax]
                        else:
                            nbrs[inbr][jax] = active_idx[jax] + switch[jax]
                    inbr += 1

            # Recompute the values of u at all Trial neighbours of Active
            # by solving the piecewise quadratic equation.
            for i in range(6):
                nbr    = nbrs[i]
                if not stencil(nbr, max_idx) or known[nbr[0], nbr[1], nbr[2]]:
                    continue
                if vv[nbr[0], nbr[1], nbr[2]] > 0:
                    for iax in range(3):
                        switch = [0, 0, 0]
                        idrxn = 0
                        if norm[nbr[0], nbr[1], nbr[2], iax] == 0:
                            aa[iax], bb[iax], cc[iax] = 0, 0, 0
                            continue
                        for idrxn in range(2):
                            switch[iax] = drxns[idrxn]
                            nbr1_i1 = (nbr[0]+switch[0]+max_idx[0]) % max_idx[0]\
                                if is_periodic[0] else nbr[0]+switch[0]
                            nbr1_i2 = (nbr[1]+switch[1]+max_idx[1]) % max_idx[1]\
                                if is_periodic[1] else nbr[1]+switch[1]
                            nbr1_i3 = (nbr[2]+switch[2]+max_idx[2]) % max_idx[2]\
                                if is_periodic[2] else nbr[2]+switch[2]
                            nbr2_i1 = (nbr[0]+2*switch[0]+max_idx[0]) % max_idx[0]\
                                if is_periodic[0] else nbr[0]+2*switch[0]
                            nbr2_i2 = (nbr[1]+2*switch[1]+max_idx[1]) % max_idx[1]\
                                if is_periodic[1] else nbr[1]+2*switch[1]
                            nbr2_i3 = (nbr[2]+2*switch[2]+max_idx[2]) % max_idx[2]\
                                if is_periodic[2] else nbr[2]+2*switch[2]
                            if (
                                (
                                   drxns[idrxn] == -1
                                   and (nbr[iax] > 1 or is_periodic[iax])
                                )
                                or
                                (
                                    drxns[idrxn] == 1
                                    and (nbr[iax] < max_idx[iax] - 2 or is_periodic[iax])
                                )
                            )\
                                and known[nbr2_i1, nbr2_i2, nbr2_i3]\
                                and known[nbr1_i1, nbr1_i2, nbr1_i3]\
                                and tt[nbr2_i1, nbr2_i2, nbr2_i3] \
                                    <= tt[nbr1_i1, nbr1_i2, nbr1_i3]\
                            :
                                order[idrxn] = 2
                                fdu[idrxn]  = drxns[idrxn] * (
                                    - 3 * tt[nbr[0], nbr[1], nbr[2]]
                                    + 4 * tt[nbr1_i1, nbr1_i2, nbr1_i3]
                                    -     tt[nbr2_i1, nbr2_i2, nbr2_i3]
                                ) / (2 * norm[nbr[0], nbr[1], nbr[2], iax])
                            elif (
                                (
                                    drxns[idrxn] == -1
                                    and (nbr[iax] > 0 or is_periodic[iax])
                                )
                                or (
                                    drxns[idrxn] ==  1
                                    and (nbr[iax] < max_idx[iax] - 1 or is_periodic[iax])
                                )
                            )\
                                and known[nbr1_i1, nbr1_i2, nbr1_i3]\
                            :
                                order[idrxn] = 1
                                fdu[idrxn] = drxns[idrxn] * (
                                    tt[nbr1_i1, nbr1_i2, nbr1_i3]
                                  - tt[nbr[0], nbr[1], nbr[2]]
                                ) / norm[nbr[0], nbr[1], nbr[2], iax]
                            else:
                                order[idrxn], fdu[idrxn] = 0, 0
                        if fdu[0] > -fdu[1]:
                            # Do the update using the backward operator
                            idrxn, switch[iax] = 0, -1
                        else:
                            # Do the update using the forward operator
                            idrxn, switch[iax] = 1, 1
                        nbr1_i1 = (nbr[0]+switch[0]+max_idx[0]) % max_idx[0]\
                            if is_periodic[0] else nbr[0]+switch[0]
                        nbr1_i2 = (nbr[1]+switch[1]+max_idx[1]) % max_idx[1]\
                            if is_periodic[1] else nbr[1]+switch[1]
                        nbr1_i3 = (nbr[2]+switch[2]+max_idx[2]) % max_idx[2]\
                            if is_periodic[2] else nbr[2]+switch[2]
                        nbr2_i1 = (nbr[0]+2*switch[0]+max_idx[0]) % max_idx[0]\
                            if is_periodic[0] else nbr[0]+2*switch[0]
                        nbr2_i2 = (nbr[1]+2*switch[1]+max_idx[1]) % max_idx[1]\
                            if is_periodic[1] else nbr[1]+2*switch[1]
                        nbr2_i3 = (nbr[2]+2*switch[2]+max_idx[2]) % max_idx[2]\
                            if is_periodic[2] else nbr[2]+2*switch[2]
                        if order[idrxn] == 2:
                            aa[iax] = 9 / (4*norm[nbr[0], nbr[1], nbr[2], iax] ** 2)
                            bb[iax] = (
                                6 * tt[nbr2_i1, nbr2_i2, nbr2_i3]
                             - 24 * tt[nbr1_i1, nbr1_i2, nbr1_i3]
                            ) / (4 * norm[nbr[0], nbr[1], nbr[2], iax]**2)
                            cc[iax] = (
                                       tt[nbr2_i1, nbr2_i2, nbr2_i3]**2
                                -  8 * tt[nbr2_i1, nbr2_i2, nbr2_i3]
                                     * tt[nbr1_i1, nbr1_i2, nbr1_i3]
                                + 16 * tt[nbr1_i1, nbr1_i2, nbr1_i3]**2
                            ) / (4 * norm[nbr[0], nbr[1], nbr[2], iax]**2)
                        elif order[idrxn] == 1:
                            aa[iax] = 1 / norm[nbr[0], nbr[1], nbr[2], iax]**2
                            bb[iax] = -2 * tt[nbr1_i1, nbr1_i2, nbr1_i3]\
                                / norm[nbr[0], nbr[1], nbr[2], iax] ** 2
                            cc[iax] = tt[nbr1_i1, nbr1_i2, nbr1_i3]**2\
                                / norm[nbr[0], nbr[1], nbr[2], iax]**2
                        elif order[idrxn] == 0:
                            aa[iax], bb[iax], cc[iax] = 0, 0, 0
                    a = aa[0] + aa[1] + aa[2]
                    if a == 0:
                        count_a += 1
                        continue
                    b = bb[0] + bb[1] + bb[2]
                    c = cc[0] + cc[1] + cc[2] - 1/vv[nbr[0], nbr[1], nbr[2]]**2
                    if b**2 < 4*a*c:
                        # This is a hack to solve the quadratic equation
                        # when the discrimnant is negative. This hack
                        # simply sets the discriminant to zero.
                        new = -b / (2*a)
                        count_b += 1
                    else:
                        new = (-b + sqrt(b**2 - 4*a*c)) / (2*a)
                    if new < tt[nbr[0], nbr[1], nbr[2]]:
                        tt[nbr[0], nbr[1], nbr[2]] = new
                        # Tag as Trial all neighbours of Active that are not
                        # Alive. If the neighbour is in Far, remove it from
                        # that list and add it to Trial.
                        if unknown[nbr[0], nbr[1], nbr[2]]:
                            trial.push(nbr[0], nbr[1], nbr[2])
                            unknown[nbr[0], nbr[1], nbr[2]] = False
                        else:
                            trial.sift_down(0, trial.heap_index[nbr[0], nbr[1], nbr[2]])
        return (True)


    cpdef np.ndarray[constants.REAL_t, ndim=2] trace_ray(
            EikonalSolver self,
            constants.REAL_t[:] end
    ):
        """
        trace_ray(self, end)

        Trace the ray ending at *end*.

        This method traces the ray that ends at *end* in reverse
        direction by taking small steps along the path of steepest
        traveltime descent. The resulting ray is reversed before being
        returned, so it is in the normal forward-time orientation.

        :param end: Coordinates of the ray's end point.
        :type end: numpy.ndarray(shape=(3,), dtype=numpy.float)

        :return: The ray path ending at *end*.
        :rtype:  numpy.ndarray(shape=(N,3), dtype=numpy.float)
        """

        cdef cpp_vector[constants.REAL_t *]       ray
        cdef constants.REAL_t                     norm, step_size
        cdef constants.REAL_t                     *point_new
        cdef constants.REAL_t[3]                  gg, point_last, point_2last
        cdef Py_ssize_t                           idx, jdx
        cdef np.ndarray[constants.REAL_t, ndim=2] ray_np
        cdef fields.VectorField3D                 grad
        cdef fields.ScalarField3D                 traveltime

        point_new = <constants.REAL_t *> malloc(3 * sizeof(constants.REAL_t))
        point_new[0], point_new[1], point_new[2] = end
        ray.push_back(point_new)
        step_size = self.step_size
        grad = self.traveltime.gradient
        # Create an interpolator for the travel-time field
        traveltime = self.traveltime
        point_last   = ray.back()
        while True:
            gg   = grad.value(point_last)
            norm = sqrt(gg[0]**2 + gg[1]**2 + gg[2]**2)
            for idx in range(3):
                gg[idx] /= norm
            if self.coord_sys == 'spherical':
                gg[1] /= point_last[0]
                gg[2] /= point_last[0] * sin(point_last[1])
            point_new = <constants.REAL_t *> malloc(3 * sizeof(constants.REAL_t))
            for idx in range(3):
                point_new[idx] = point_last[idx] - step_size * gg[idx]
            point_2last = ray.back()
            ray.push_back(point_new)
            point_last  = ray.back()
            try:
                if traveltime.value(point_2last) <= traveltime.value(point_last):
                    break
            except ValueError:
                for idx in range(ray.size()-1):
                    free(ray[idx])
                return (None)
        ray_np = np.zeros((ray.size()-1, 3), dtype=constants.DTYPE_REAL)
        for idx in range(ray.size()-1):
            for jdx in range(3):
                ray_np[idx, jdx] = ray[idx][jdx]
            free(ray[idx])
        return (np.flipud(ray_np))




cdef inline bint stencil(Py_ssize_t[:] idx, Py_ssize_t[:] max_idx):
    return (
            (idx[0] >= 0)
        and (idx[0] < max_idx[0])
        and (idx[1] >= 0)
        and (idx[1] < max_idx[1])
        and (idx[2] >= 0)
        and (idx[2] < max_idx[2])
    )
