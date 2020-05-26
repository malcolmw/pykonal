import h5py
import numpy as np
import os

from . import fields


class TraveltimeInventory(object):

    def __init__(self, path, mode="r"):
        self._mode = mode
        self._path = path
        self._f5 = h5py.File(path, mode=mode)

    def __del__(self):
        self.f5.close()

    def __enter__(self):
        return (self)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__del__()

    @property
    def f5(self):
        return (self._f5)

    @property
    def mode(self):
        return (self._mode)

    @mode.setter
    def mode(self, value):
        self._mode = value
        self.f5.close()
        self._f5 = h5py.File(self.path, mode=value)

    @property
    def path(self):
        return (self._path)


    def add(self, field, key):

        group = self.f5.create_group(key)
        group.attrs["coord_sys"] = field.coord_sys
        group.attrs["field_type"] = field.field_type

        for attr in ("min_coords", "node_intervals", "npts", "values"):
            group.create_dataset(attr, data=getattr(field, attr))

        return (True)


    def merge(self, paths):

        for path in paths:

            print (f"Merging {path}")

            _, filename = os.path.split(path)
            filename, file_ext = os.path.splitext(filename)
            network, station, phase = filename.split(".")
            field = fields.read_hdf(path)
            self.add(field, "/".join([network, station, phase]))

        return (True)


    def read(self, key, min_coords=None, max_coords=None):

        group = self.f5[key]

        _coord_sys = group.attrs["coord_sys"]
        _field_type = group.attrs["field_type"]
        _min_coords = group["min_coords"][:]
        _node_intervals = group["node_intervals"][:]
        _npts = group["npts"][:]

        if min_coords is not None:
            min_coords = np.array(min_coords)

        if max_coords is not None:

            max_coords = np.array(max_coords)

        if min_coords is not None and max_coords is not None:

            if np.any(min_coords >= max_coords):

                raise(ValueError("All values of min_coords must satisfy min_coords < max_coords."))

        if min_coords is not None:

            idx_start = (min_coords - _min_coords) / _node_intervals
            idx_start = np.floor(idx_start)
            idx_start = idx_start.astype(np.int32)
            idx_start = np.clip(idx_start, 0, _npts - 1)

        else:

            idx_start = np.array([0, 0, 0])

        if max_coords is not None:

            idx_end = (max_coords - _min_coords) / _node_intervals
            idx_end = np.ceil(idx_end) + 1
            idx_end = idx_end.astype(np.int32)
            idx_end = np.clip(idx_end, idx_start + 1, _npts)

        else:

            idx_end = _npts

        if _field_type == "scalar":
            field = fields.ScalarField3D(coord_sys=_coord_sys)
        elif _field_type == "vector":
            field = fields.VectorField3D(coord_sys=_coord_sys)
        else:
            raise (ValueError(f"Unrecognized field type: {_field_type}"))

        field.min_coords = _min_coords  +  idx_start * _node_intervals
        field.node_intervals = _node_intervals
        field.npts = idx_end - idx_start
        idxs = tuple(slice(idx_start[idx], idx_end[idx]) for idx in range(3))
        field.values = group["values"][idxs]

        return (field)
