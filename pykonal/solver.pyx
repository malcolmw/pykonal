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
cimport cython
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
        self.cy_coord_sys = coord_sys
        self.cy_velocity = fields.ScalarField3D(coord_sys=self.coord_sys)


    @property
    def trial(self):
        """
        [*Read/Write*, :class:`pykonal.heapq.Heap`] Heap of node
        indices in *Trial*.
        """
        if self.cy_trial is None:
            self.cy_trial = heapq.Heap(self.traveltime.values)
        return (self.cy_trial)

    @property
    def coord_sys(self):
        """
        [*Read only*, :class:`str`] Coordinate system of solver
        {"Cartesian", "spherical"}.
        """
        return (self.cy_coord_sys)


    @property
    def known(self):
        """
        [*Read/Write*, :class:`numpy.ndarray`\ (shape=(N0,N1,N2), dtype=numpy.bool)] 3D array of booleans
        indicating whether nodes are in *Known*.
        """
        try:
            return (np.asarray(self.cy_known))
        except AttributeError:
            self.cy_known = np.zeros(self.tt.npts, dtype=constants.DTYPE_BOOL)
        return (np.asarray(self.cy_known))

    @property
    def unknown(self):
        """
        [*Read/Write*, :class:`numpy.ndarray`\ (shape=(N0,N1,N2), dtype=numpy.bool)] 3D array of booleans
        indicating whether nodes are in *Unknown*.
        """
        try:
            return (np.asarray(self.cy_unknown))
        except AttributeError:
            self.cy_unknown = np.ones(self.tt.npts, dtype=constants.DTYPE_BOOL)
        return (np.asarray(self.cy_unknown))



    @property
    def norm(self):
        """
        [*Read-only*, :class:`numpy.ndarray`\ (shape=(N0,N1,N2,3), dtype=numpy.float)] 4D array of scaling
        factors for gradient operator.
        """
        try:
            return (np.asarray(self.cy_norm))
        except AttributeError:
            norm = np.tile(
                self.traveltime.node_intervals,
                np.append(self.traveltime.npts, 1)
            )
            if self.coord_sys == 'spherical':
                norm[..., 1] *= self.traveltime.nodes[..., 0]
                norm[..., 2] *= self.traveltime.nodes[..., 0]
                norm[..., 2] *= np.sin(self.traveltime.nodes[..., 1])
            self.cy_norm = norm
        return (np.asarray(self.cy_norm))

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
        if self.cy_traveltime is None:
            self.cy_traveltime = fields.ScalarField3D(coord_sys=self.coord_sys)
            self.cy_traveltime.min_coords = self.velocity.min_coords
            self.cy_traveltime.node_intervals = self.velocity.node_intervals
            self.cy_traveltime.npts = self.velocity.npts
            self.cy_traveltime.values = np.full_like(self.velocity.values, fill_value=np.inf)
        return (self.cy_traveltime)

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
        return (self.cy_velocity)

    @property
    def vv(self):
        """
        [*Read/Write*, :class:`pykonal.fields.ScalarField3D`] Alias for
        self.velocity.
        """
        return (self.velocity)

    @cython.initializedcheck(False)
    cpdef constants.BOOL_t solve(EikonalSolver self):
        """
        solve(self)

        Solve the Eikonal equation using the FMM.

        :return: Returns True upon successful execution.
        :rtype:  bool
        """
        cdef Py_ssize_t                           i, iax, idrxn, iheap
        cdef Py_ssize_t[6][3]                     nbrs
        cdef Py_ssize_t[3]                        nbr, switch, max_idx, active_idx
        cdef Py_ssize_t[2]                        drxns = [-1, 1]
        cdef Py_ssize_t[:,:,:]                    heap_index
        cdef (Py_ssize_t, Py_ssize_t, Py_ssize_t) idx
        cdef int                                  count_a = 0
        cdef int                                  count_b = 0
        cdef int                                  inbr
        cdef int[2]                               order
        cdef constants.REAL_t                     a, b, c, bfd, ffd, new
        cdef constants.REAL_t[2]                  fdu
        cdef constants.REAL_t[3]                  aa, bb, cc
        cdef constants.REAL_t[:,:,:]              tt, vv
        cdef constants.REAL_t[:,:,:,:]            norm
        cdef constants.BOOL_t[3]                  iax_isperiodic,
        cdef constants.BOOL_t[:,:,:]              known, unknown
        cdef heapq.Heap                           trial

        for iax in range(3):
            max_idx[iax] = <Py_ssize_t> self.cy_traveltime.cy_npts[iax]
            iax_isperiodic[iax] = <constants.BOOL_t> self.cy_traveltime.cy_iax_isperiodic[iax]

        tt = self.traveltime.values
        vv = self.velocity.values
        norm = self.norm
        known = self.known
        unknown = self.unknown
        trial = self.trial
        heap_index = trial.cy_heap_index


        while trial.cy_keys.size() > 0:
            # Let Active be the point in Trial with the smallest
            # traveltime value.
            idx = trial.pop()
            active_idx = [idx[0], idx[1], idx[2]]
            known[active_idx[0], active_idx[1], active_idx[2]] = True

            # Determine the indices of neighbouring nodes.
            inbr = 0
            for iax in range(3):
                switch = [0, 0, 0]
                for idrxn in range(2):
                    switch[iax] = drxns[idrxn]
                    for jax in range(3):
                        if iax_isperiodic[jax]:
                            nbrs[inbr][jax] = (
                                  active_idx[jax]
                                + switch[jax]
                                + max_idx[jax]
                            )\
                            % max_idx[jax]
                        else:
                            nbrs[inbr][jax] = active_idx[jax] + switch[jax]
                    inbr += 1

            # Recompute the traveltime values at all Trial neighbours
            # of Active by solving the piecewise quadratic equation.
            for i in range(6):
                nbr    = nbrs[i]
                if not stencil(nbr[0], nbr[1], nbr[2], max_idx[0], max_idx[1], max_idx[2]) or known[nbr[0], nbr[1], nbr[2]]:
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
                                if iax_isperiodic[0] else nbr[0]+switch[0]
                            nbr1_i2 = (nbr[1]+switch[1]+max_idx[1]) % max_idx[1]\
                                if iax_isperiodic[1] else nbr[1]+switch[1]
                            nbr1_i3 = (nbr[2]+switch[2]+max_idx[2]) % max_idx[2]\
                                if iax_isperiodic[2] else nbr[2]+switch[2]
                            nbr2_i1 = (nbr[0]+2*switch[0]+max_idx[0]) % max_idx[0]\
                                if iax_isperiodic[0] else nbr[0]+2*switch[0]
                            nbr2_i2 = (nbr[1]+2*switch[1]+max_idx[1]) % max_idx[1]\
                                if iax_isperiodic[1] else nbr[1]+2*switch[1]
                            nbr2_i3 = (nbr[2]+2*switch[2]+max_idx[2]) % max_idx[2]\
                                if iax_isperiodic[2] else nbr[2]+2*switch[2]
                            if (
                                (
                                   drxns[idrxn] == -1
                                   and (nbr[iax] > 1 or iax_isperiodic[iax])
                                )
                                or
                                (
                                    drxns[idrxn] == 1
                                    and (nbr[iax] < max_idx[iax] - 2 or iax_isperiodic[iax])
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
                                    and (nbr[iax] > 0 or iax_isperiodic[iax])
                                )
                                or (
                                    drxns[idrxn] ==  1
                                    and (nbr[iax] < max_idx[iax] - 1 or iax_isperiodic[iax])
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
                            if iax_isperiodic[0] else nbr[0]+switch[0]
                        nbr1_i2 = (nbr[1]+switch[1]+max_idx[1]) % max_idx[1]\
                            if iax_isperiodic[1] else nbr[1]+switch[1]
                        nbr1_i3 = (nbr[2]+switch[2]+max_idx[2]) % max_idx[2]\
                            if iax_isperiodic[2] else nbr[2]+switch[2]
                        nbr2_i1 = (nbr[0]+2*switch[0]+max_idx[0]) % max_idx[0]\
                            if iax_isperiodic[0] else nbr[0]+2*switch[0]
                        nbr2_i2 = (nbr[1]+2*switch[1]+max_idx[1]) % max_idx[1]\
                            if iax_isperiodic[1] else nbr[1]+2*switch[1]
                        nbr2_i3 = (nbr[2]+2*switch[2]+max_idx[2]) % max_idx[2]\
                            if iax_isperiodic[2] else nbr[2]+2*switch[2]
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
                            trial.sift_down(0, heap_index[nbr[0], nbr[1], nbr[2]])
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
        cdef constants.REAL_t                     norm, step_size, value, value_1back
        cdef constants.REAL_t                     *point_new
        cdef constants.REAL_t[3]                  gg, point, point_1back
        cdef Py_ssize_t                           idx, jdx
        cdef np.ndarray[constants.REAL_t, ndim=2] ray_np
        cdef fields.VectorField3D                 grad
        cdef fields.ScalarField3D                 traveltime
        cdef str                                  coord_sys


        coord_sys = self.coord_sys
        step_size = self.step_size
        grad = self.traveltime.gradient
        traveltime = self.traveltime

        point_new = <constants.REAL_t *> malloc(3 * sizeof(constants.REAL_t))
        point_new[0], point_new[1], point_new[2] = end
        ray.push_back(point_new)
        point = ray.back()
        value = traveltime.value(point)

        while True:
            gg   = grad.value(point)
            norm = sqrt(gg[0]**2 + gg[1]**2 + gg[2]**2)
            for idx in range(3):
                gg[idx] /= norm
            if coord_sys == "spherical":
                gg[1] /= point[0]
                gg[2] /= point[0] * sin(point[1])
            point_new = <constants.REAL_t *> malloc(3 * sizeof(constants.REAL_t))
            for idx in range(3):
                point_new[idx] = point[idx] - step_size * gg[idx]
            point_1back = ray.back()
            ray.push_back(point_new)
            point  = ray.back()
            try:
                value_1back = value
                value = traveltime.value(point)
                if value_1back <= value or np.isnan(value):
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


class PointSourceSolver(EikonalSolver):
    """
    A convenience class to improve precision when solving the common
    problem of a point source.

    This class implements a pair of complementary computational grids
    to improve precision. A refined spherical grid centered on the
    source is used in the near-source region. The user can specify
    whether a Cartesian or spherical grid is used in the far field.
    The refinement of the near-field grid will be handled automatically
    using some reasonable values if the user does not manually configure
    it.
    """
    def __init__(self, coord_sys="cartesian"):
        super(PointSourceSolver, self).__init__(coord_sys=coord_sys)

    @property
    def dphi(self):
        """
        [*Read/Write*, float] Node interval (in radians) along the
        azimuthal axis of the refined near-field grid.
        """
        if not hasattr(self, "_dphi"):
            self._dphi = (2 * np.pi) / self.nphi
        return (self._dphi)

    @dphi.setter
    def dphi(self, value):
        self._dphi = value

    @property
    def drho(self):
        """
        [*Read/Write*, float] Node interval (in arbitrary distance units)
        along the radial axis of the refined near-field grid.
        """
        if not hasattr(self, "_drho"):
            if self.coord_sys == "cartesian":
                self._drho = self.vv.node_intervals.min() / 8
            else:
                self._drho = self.vv.node_intervals[0] / 8
        return (self._drho)

    @drho.setter
    def drho(self, value):
        self._drho = value

    @property
    def dtheta(self):
        """
        [*Read/Write*, float] Node interval (in radians) along the
        polar axis of the refined near-field grid.
        """
        if not hasattr(self, "_dtheta"):
            self._dtheta = np.pi / (self.ntheta - 1)
        return (self._dtheta)

    @dtheta.setter
    def dtheta(self, value):
        self._dtheta = value

    @property
    def nphi(self):
        """
        [*Read*, int] Number of grid nodes along the azimuthal axis
        of the refined near-field grid.
        """
        if not hasattr(self, "_nphi"):
            self._nphi = 64
        return (self._nphi)

    @property
    def nrho(self):
        """
        [*Read/Write*, int] Number of grid nodes along the radial axis
        of the refined near-field grid.
        """
        if not hasattr(self, "_nrho"):
            self._nrho = 64
        return (self._nrho)

    @nrho.setter
    def nrho(self, value):
        self._nrho = value

    @property
    def ntheta(self):
        """
        [*Read*, int] Number of grid nodes along the polar axis
        of the refined near-field grid.
        """
        if not hasattr(self, "_ntheta"):
            self._ntheta = 33
        return (self._ntheta)

    @property
    def near_field(self):
        """
        [*Read/Write*, :class:`EikonalSolver`] Solver for the Eikonal
        equation in the near-field region.
        """
        if not hasattr(self, "_near_field"):
            self._near_field = EikonalSolver(coord_sys="spherical")
        return (self._near_field)

    @property
    def src_loc(self):
        """
        [*Read/Write*, (float, float, float)] Location of the point
        source in the coordinates of the far-field grid.
        """
        return (self._src_loc)

    @src_loc.setter
    def src_loc(self, value):
        self._src_loc = value

    def initialize_near_field_grid(self) -> bool:
        """
        Initialize the near-field grid.

        :return: Returns True upon successful completion.
        :rtype: bool
        """
        # TODO:: rho0 should be be the smallest non-zero value between
        # The distance to the closest node and drho.
        self.near_field.vv.min_coords = self.drho, 0, 0
        self.near_field.vv.node_intervals = self.drho, self.dtheta, self.dphi
        self.near_field.vv.npts = self.nrho, self.ntheta, self.nphi
        return (True)


    def initialize_near_field_narrow_band(self) -> bool:
        """
        Initialize the narrow band of the near-field grid using the
        layer of nodes closest to the source.

        :return: Returns True upon successful completion.
        :rtype: bool
        """
        for it in range(self.ntheta):
            for ip in range(self.nphi):
                idx = (0, it, ip)
                vv = self.near_field.vv.values[idx]
                if not np.isnan(vv):
                    self.near_field.tt.values[idx] = self.drho / vv
                    self.near_field.unknown[idx] = False
                    self.near_field.trial.push(*idx)


    def initialize_far_field_narrow_band(self) -> bool:
        """
        Initialize the narrow band of the far-field grid using all the
        nodes with finite traveltimes.

        :return: Returns True upon successful completion.
        :rtype: bool
        """
        # Update the FMM state variables.
        for idx in np.argwhere(~np.isinf(self.tt.values)):
            idx = tuple(idx)
            self.unknown[idx] = False
            self.trial.push(*idx)


    def interpolate_near_field_traveltime_onto_far_field(self) -> bool:
        """
        Interpolate the near-field traveltime values onto the far-field
        grid.

        :return: Returns True upon successful completion.
        :rtype: bool
        """
        # Transform the coordinates of the far-field grid nodes to the
        # near-field coordinate system.
        nodes = self.vv.transform_coordinates("spherical", self.src_loc)
        # Find the indices of the far-field grid nodes that are inside
        # the near-field grid.
        bool_idx = True
        for iax in range(3):
            bool_idx = bool_idx &(
                 (self.near_field.vv.iax_isnull[iax])
                |(self.near_field.vv.iax_isperiodic[iax])
                |(
                     (nodes[...,iax] >= self.near_field.vv.min_coords[iax])
                    &(nodes[...,iax] <= self.near_field.vv.max_coords[iax])
                )
            )
        idxs = np.nonzero(bool_idx)
        # Sample the near-field taveltime field on the far-field nodes.
        # Make sure to filter out any NaN values.
        tt = self.near_field.tt.resample(nodes[idxs].reshape(-1, 3))
        idxs = np.swapaxes(np.stack(idxs), 0, 1)[~np.isnan(tt)]
        self.tt.values[(idxs[:,0], idxs[:,1], idxs[:,2])] = tt[~np.isnan(tt)]


    def interpolate_far_field_velocity_onto_near_field(self) -> bool:
        """
        Interpolate the far-field velocity model onto the near-field
        grid.

        :return: Returns True upon successful completion.
        :rtype: bool
        """
        # Compute the coordinates of the origin of the far-field grid
        # with respect to the near-field coordinate system.
        if self.coord_sys == "cartesian":
            r0 = np.sqrt(np.sum(np.square(self.src_loc)))
            t0 = np.pi - np.arccos(self.src_loc[2]/ r0)
            p0 = (np.arctan2(self.src_loc[1], self.src_loc[0]) + np.pi) % (2 * np.pi)
        else:
            r0 = self.src_loc[0]
            t0 = np.pi - self.src_loc[1]
            p0 = (self.src_loc[2] + np.pi) % (2 * np.pi)
        origin = (r0, t0, p0)
        # Transform the coordinates of the near-field grid nodes to the
        # far-field coordinate system.
        nodes = self.near_field.vv.transform_coordinates(self.coord_sys, origin)
        # Find the indices of the near-field grid nodes that are inside
        # the far-field grid.
        bool_idx = True
        for iax in range(3):
            bool_idx = bool_idx &(
                 (self.vv.iax_isnull[iax])
                |(self.vv.iax_isperiodic[iax])
                |(
                     (nodes[...,iax] >= self.vv.min_coords[iax])
                    &(nodes[...,iax] <= self.vv.max_coords[iax])
                )
            )
        idxs = np.nonzero(bool_idx)
        # Sample the far-field velocity model on the near-field nodes.
        self.near_field.vv.values[idxs] = self.vv.resample(nodes[idxs].reshape(-1, 3))
        return (True)


    def solve(self):
        """
        Solve the Eikonal equation on the far-field grid using the
        refined source grid in the near-field region.

        :return: Returns True upon successful completion.
        :rtype: bool
        """
        # Initialize the near-field grid.
        self.initialize_near_field_grid()
        # Interpolate far-field velocity model onto near-field nodes.
        self.interpolate_far_field_velocity_onto_near_field()
        # Initialize the narrow band of the near-field grid.
        self.initialize_near_field_narrow_band()
        # Propagate the wavefront through the near field.
        self.near_field.solve()
        # Interpolate the near-field traveltime values onto the far-field
        # grid.
        self.interpolate_near_field_traveltime_onto_far_field()
        # Initialize the narrow band of the far-field grid.
        self.initialize_far_field_narrow_band()
        # Propagate the wavefront through the far field.
        super(PointSourceSolver, self).solve()
        return (True)




cdef inline bint stencil(
        Py_ssize_t idx0, Py_ssize_t idx1, Py_ssize_t idx2,
        Py_ssize_t max_idx0, Py_ssize_t max_idx1, Py_ssize_t max_idx2
):
    return (
            (idx0 >= 0)
        and (idx0 < max_idx0)
        and (idx1 >= 0)
        and (idx1 < max_idx1)
        and (idx2 >= 0)
        and (idx2 < max_idx2)
    )
