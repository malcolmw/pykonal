# Cython compiler directives.
# distutils: language=c++
# cython: profile=True


import numpy as np
import os
import pykonal
import scipy.optimize
import tempfile

from . import constants as _constants
from . import inventory as _inventory
from . import solver as _solver
from . import transformations as _transformations

cimport numpy as np

from libc.math cimport sqrt

from . cimport fields
from . cimport constants

inf = np.inf

cdef class EQLocator(object):
    """
    EQLocator(stations, tt_dir=None)

    A class to locate earthquakes.
    """
    def __init__(
        self,
        stations: dict,
        traveltime_inventory: str,
        coord_sys: str="spherical"
    ):
        self.cy_arrivals = {}
        self.cy_traveltimes = {}
        self.cy_coord_sys = coord_sys
        self.cy_stations = stations
        inventory = _inventory.TraveltimeInventory(traveltime_inventory, mode="r")
        self.cy_traveltime_inventory = inventory


    def __del__(self):
        self.traveltime_inventory.f5.close()


    def __enter__(self):
        return (self)


    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__del__()


    cpdef constants.BOOL_t add_arrivals(EQLocator self, dict arrivals):
        self.cy_arrivals = {**self.cy_arrivals, **arrivals}
        return (True)


    cpdef constants.BOOL_t add_residual_rvs(EQLocator self, dict residual_rvs):
        self.cy_residual_rvs = {**self.cy_residual_rvs, **residual_rvs}
        return (True)


    cpdef constants.BOOL_t clear_arrivals(EQLocator self):
        self.cy_arrivals = {}
        return (True)


    cpdef constants.BOOL_t clear_residual_rvs(EQLocator self):
        self.cy_residual_rvs = {}
        return (True)


    @property
    def arrivals(self) -> dict:
        return (self.cy_arrivals)

    @arrivals.setter
    def arrivals(self, value: dict):
        self.cy_arrivals = value

    @property
    def coord_sys(self) -> str:
        return (self.cy_coord_sys)

    @property
    def grid(self) -> object:
        if self.cy_grid is None:
            self.cy_grid = fields.ScalarField3D(coord_sys=self.cy_coord_sys)
        return (self.cy_grid)

    @property
    def stations(self) -> dict:
        return (self.cy_stations)

    @stations.setter
    def stations(self, value: dict):
        self.cy_stations = value

    @property
    def traveltime_inventory(self) -> object:
        return (self.cy_traveltime_inventory)

    @property
    def pwave_velocity(self) -> object:
        if self.cy_pwave_velocity is None:
            self.cy_pwave_velocity = fields.ScalarField3D(coord_sys=self.cy_coord_sys)
            self.cy_pwave_velocity.min_coords = self.cy_grid.min_coords
            self.cy_pwave_velocity.node_intervals = self.cy_grid.node_intervals
            self.cy_pwave_velocity.npts = self.cy_grid.npts
        return (self.cy_pwave_velocity)

    @pwave_velocity.setter
    def pwave_velocity(self, value: np.ndarray):
        if self.cy_pwave_velocity is None:
            self.pwave_velocity
        self.cy_pwave_velocity.values = value

    @property
    def vp(self) -> object:
        return (self.pwave_velocity)

    @vp.setter
    def vp(self, value: np.ndarray):
        self.pwave_velocity = value

    @property
    def residual_rvs(self) -> dict:
        return (self.cy_residual_rvs)

    @residual_rvs.setter
    def residual_rvs(self, value: dict):
        self.cy_residual_rvs = value

    @property
    def swave_velocity(self) -> object:
        if self.cy_swave_velocity is None:
            self.cy_swave_velocity = fields.ScalarField3D(coord_sys=self.cy_coord_sys)
            self.cy_swave_velocity.min_coords = self.cy_grid.min_coords
            self.cy_swave_velocity.node_intervals = self.cy_grid.node_intervals
            self.cy_swave_velocity.npts = self.cy_grid.npts
        return (self.cy_swave_velocity)

    @swave_velocity.setter
    def swave_velocity(self, value: np.ndarray):
        if self.cy_swave_velocity is None:
            self.swave_velocity
        self.cy_swave_velocity.values = value

    @property
    def traveltimes(self) -> dict:
        return (self.cy_traveltimes)

    @traveltimes.setter
    def traveltimes(self, value: dict):
        self.cy_traveltimes = value

    @property
    def vs(self) -> object:
        return (self.swave_velocity)

    @vs.setter
    def vs(self, value: np.ndarray):
        self.swave_velocity = value


    cpdef constants.BOOL_t read_traveltimes(
        EQLocator self,
        constants.REAL_t[:] min_coords=None,
        constants.REAL_t[:] max_coords=None
    ):

        inventory = self.cy_traveltime_inventory
        self.cy_traveltimes = {
            index: inventory.read(
                "/".join(index),
                min_coords=min_coords,
                max_coords=max_coords
            ) for index in self.cy_arrivals
        }

        return (True)


    #cpdef np.ndarray[constants.REAL_t, ndim=1] grid_search(EQLocator self):
    #    values = [self.cy_arrivals[key]-np.ma.masked_invalid(self.cy_traveltimes[key].values) for key in self.cy_traveltimes]
    #    values = np.stack(values)
    #    std = values.std(axis=0)
    #    arg_min = np.argmin(std)
    #    idx_min = np.unravel_index(arg_min, std.shape)
    #    coords = self.cy_grid.nodes[idx_min]
    #    time = values.mean(axis=0)[idx_min]
    #    return (np.array([*coords, time], dtype=_constants.DTYPE_REAL))

    #
    cpdef constants.REAL_t rms(EQLocator self, constants.REAL_t[:] hypocenter):
        cdef tuple                   key
        cdef dict                    arrivals
        cdef dict                    traveltimes
        cdef constants.REAL_t        csum = 0
        cdef constants.REAL_t        num
        cdef constants.REAL_t        t0

        arrivals = self.cy_arrivals
        traveltimes = self.cy_traveltimes
        t0 = hypocenter[3]

        for key in arrivals:
            num = arrivals[key] - t0
            num -= traveltimes[key].value(hypocenter[:3], null=np.inf)
            csum += num * num

        return (sqrt(csum/len(arrivals)))


    cpdef np.ndarray[constants.REAL_t, ndim=1] locate(
        EQLocator self,
        np.ndarray[constants.REAL_t, ndim=1] initial,
        np.ndarray[constants.REAL_t, ndim=1] delta
    ):
        """
        Locate event using a grid search and Differential Evolution
        Optimization to minimize the residual RMS.
        """

        min_coords = initial - delta
        max_coords = initial + delta
        bounds = np.stack([min_coords, max_coords]).T

        self.read_traveltimes(
            min_coords=min_coords[:3],
            max_coords=max_coords[:3]
        )

        soln = scipy.optimize.differential_evolution(self.rms, bounds)

        return (soln.x)

    #cpdef constants.REAL_t log_likelihood(
    #    EQLocator self,
    #    constants.REAL_t[:] model
    #):
    #    cdef constants.REAL_t   t_pred, residual
    #    cdef constants.REAL_t   log_likelihood = 0.0
    #    cdef tuple              key
    #    cdef EQLocator[:]       junk

    #    for key in self.cy_arrivals:
    #        t_pred = model[3] + self.cy_traveltimes[key].value(model[:3])
    #        residual = self.cy_arrivals[key] - t_pred
    #        log_likelihood = log_likelihood + self.cy_residual_rvs[key].logpdf(residual)
    #    return (log_likelihood)
