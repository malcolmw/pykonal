# Cython compiler directives.
# distutils: language=c++


import numpy as np
import os
import pykonal
import scipy.optimize
import tempfile

from . import constants as _constants
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
    def __init__(self, stations: dict, coord_sys: str="spherical", tt_dir: str=None):
        self.cy_arrivals = {}
        self.cy_traveltimes = {}
        self.cy_coord_sys = coord_sys
        self.cy_stations = stations
        if tt_dir is None:
            self.cy_tempdir_obj = tempfile.TemporaryDirectory()
            self.cy_tt_dir = self.cy_tempdir_obj.name
        else:
            self.cy_tempdir_obj = None
            self.cy_tt_dir = tt_dir

    def __enter__(self):
        return (self)
    
    def __del__(self):
        self.__exit__(None, None, None)

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
        

    cpdef constants.BOOL_t add_arrivals(EQLocator self, dict arrivals):
        self.cy_arrivals = {**self.cy_arrivals, **arrivals}
        return (True)


    cpdef constants.BOOL_t cleanup(EQLocator self):
        # If the traveltime directory is temporary, clean it up.
        if self.cy_tempdir_obj is not None:
            self.cy_tempdir_obj.cleanup()
        return (True)
    

    cpdef constants.BOOL_t clear_arrivals(EQLocator self):
        self.cy_arrivals = {}
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
    def tt_dir(self) -> object:
        return (self.cy_tt_dir)
    
    @property
    def tt_dir_is_temp(self) -> bool:
        return (self.cy_tt_dir_is_temp)

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
        

    cpdef constants.BOOL_t compute_traveltime_lookup_table(EQLocator self, str station_id, str phase):
        cdef fields.ScalarField3D velocity
        if phase.upper() == "P":
            velocity = self.cy_pwave_velocity
        elif phase.upper() == "S":
            velocity = self.cy_swave_velocity
        else:
            raise (ValueError(f"Unrecognized phase {phase}"))
        solver = _solver.PointSourceSolver(coord_sys=self.coord_sys)
        solver.velocity.min_coords = velocity.cy_min_coords
        solver.velocity.node_intervals = velocity.cy_node_intervals
        solver.velocity.npts = velocity.cy_npts
        solver.velocity.values = velocity.cy_values
        solver.src_loc = self.cy_stations[station_id]
        solver.solve()
        fname = os.path.join(self.cy_tt_dir, f"{station_id}.{phase}.npz")
        solver.traveltime.savez(fname)
            
    cpdef constants.BOOL_t compute_all_traveltime_lookup_tables(EQLocator self, str phase):
        cdef tuple station_id

        for station_id in self.stations:
            self.compute_traveltime_lookup_table(station_id, phase)
        return (True)
            
    cpdef constants.BOOL_t load_traveltimes(EQLocator self):
        self.cy_traveltimes = {
            **{key: self.cy_traveltimes[key] for key in self.cy_arrivals if key in self.cy_traveltimes},
            **{key: fields.load(os.path.join(self.cy_tt_dir, f"{'.'.join(key)}.npz")) for key in self.cy_arrivals if key not in self.cy_traveltimes}
        }
        return (True)
    
    cpdef np.ndarray[constants.REAL_t, ndim=1] grid_search(EQLocator self):
        values = [self.cy_arrivals[key]-np.ma.masked_invalid(self.cy_traveltimes[key].values) for key in self.cy_traveltimes]
        values = np.stack(values)
        std = values.std(axis=0)
        arg_min = np.argmin(std)
        idx_min = np.unravel_index(arg_min, std.shape)
        coords = self.cy_grid.nodes[idx_min]
        time = values.mean(axis=0)[idx_min]
        return (np.array([*coords, time], dtype=_constants.DTYPE_REAL))
    
    cpdef constants.REAL_t cost(EQLocator self, constants.REAL_t[:] hypocenter):
        cdef tuple                   key
        cdef Py_ssize_t              idx
        cdef constants.REAL_t        csum = 0
        cdef constants.REAL_t        num
        cdef constants.REAL_t        rho, theta, phi, time
        cdef constants.REAL_t[:]     arrivals
        cdef list                    traveltimes

        arrivals = self.cy_arrivals_sorted
        traveltimes = self.cy_traveltimes_sorted

        rho = hypocenter[0]
        theta = hypocenter[1]
        phi = hypocenter[2]
        time = hypocenter[3]
        if not (
            rho > self.cy_grid.cy_min_coords[0]
            and rho < self.cy_grid.cy_max_coords[0]
            and theta > self.cy_grid.cy_min_coords[1]
            and theta < self.cy_grid.cy_max_coords[1]
            and phi > self.cy_grid.cy_min_coords[2]
            and phi < self.cy_grid.cy_max_coords[2]
        ):
            return (inf)
        for idx in range(len(arrivals)):
            num = arrivals[idx] - time
            num -= traveltimes[idx].value(hypocenter[:3])
            csum += num * num
        return (sqrt(csum/len(arrivals)))

    
    cpdef np.ndarray[constants.REAL_t, ndim=1] locate(EQLocator self):
        cdef constants.REAL_t[4] h0
        self.cy_arrivals_sorted = np.array([self.cy_arrivals[key] for key in sorted(self.cy_arrivals)])
        self.cy_traveltimes_sorted = [self.cy_traveltimes[key] for key in sorted(self.cy_arrivals)]
        h0 = self.grid_search()
        dx = np.radians(0.1)
        soln = scipy.optimize.differential_evolution(
            self.cost,
            #((h0[0]-0.1, h0[0]+0.1), (h0[1]-0.1, h0[1]+0.1), (0, 30), (h0[3]-5, h0[3]+5))
            ((6341., 6371.), (h0[1]-dx, h0[1]+dx), (h0[2]-dx, h0[2]+dx), (h0[3]-5, h0[3]+5))
        )
        return (soln.x)
