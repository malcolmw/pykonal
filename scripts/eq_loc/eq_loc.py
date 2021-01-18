"""
An MPI script to locate earthquakes.

.. author: Malcolm C. A. White
"""

import argparse
import configparser
import logging
import mpi4py.MPI as MPI
import numpy as np
import pandas as pd
import pathlib
import pykonal
import signal

PROCESSOR_NAME      = MPI.Get_processor_name()
COMM                = MPI.COMM_WORLD
WORLD_SIZE          = COMM.Get_size()
RANK                = COMM.Get_rank()
ROOT_RANK           = 0
ID_REQUEST_TAG      = 100
ID_TRANSMISSION_TAG = 101

DTYPES = {
    "latitude":  np.float64,
    "longitude": np.float64,
    "depth":     np.float64,
    "time":      np.float64,
    "residual":  np.float64,
    "event_id":  np.int64
}

ARRIVAL_FIELDS_INPUT = [
    "network",
    "station",
    "phase",
    "time",
    "event_id",
    "weight"
]
ARRIVAL_FIELDS_OUTPUT = [*ARRIVAL_FIELDS_INPUT, "residual"]
EVENT_FIELDS_INPUT = ["latitude", "longitude", "depth", "time", "event_id"]
EVENT_FIELDS_OUTPUT = [*EVENT_FIELDS_INPUT, "residual"]
STATION_FIELDS_INPUT = [
    "network",
    "station",
    "latitude",
    "longitude",
    "elevation"
]

geo2sph = pykonal.transformations.geo2sph
sph2geo = pykonal.transformations.sph2geo

def parse_args():
    """
    Parse and return command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "catalog",
        type=str,
        help="Input catalog."
    )
    parser.add_argument(
        "network_geometry",
        type=str,
        help="Input file containing network geometry (station locations)."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Output file containing event locations."
    )
    parser.add_argument(
        "-l",
        "--log_file",
        type=str,
        default="eq_loc.log",
        help="Log file."
    )
    parser.add_argument(
        "-c",
        "--configuration_file",
        type=str,
        default="eq_loc.cfg",
        help="Configuration file."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Be verbose."
    )

    return (parser.parse_args())


def parse_cfg(configuration_file):
    """
    Parse and return contents of the configuration file.
    """

    cfg = dict()
    parser = configparser.ConfigParser()
    parser.read(configuration_file)
    cfg["traveltime_inventory"] = parser.get(
        "DEFAULT",
        "traveltime_inventory",
        fallback=None
    )
    cfg["dlat"] = parser.getfloat(
        "Differential Evolution",
        "dlat",
        fallback=0.1
    )
    cfg["dlon"] = parser.getfloat(
        "Differential Evolution",
        "dlon",
        fallback=0.1
    )
    cfg["dz"] = parser.getfloat(
        "Differential Evolution",
        "dz",
        fallback=10
    )
    cfg["dt"] = parser.getfloat(
        "Differential Evolution",
        "dt",
        fallback=10
    )
    for key in cfg:
        if cfg[key] == "None":
            cfg[key] = None

    return (cfg)


def main(argc, cfg):
    """
    This is the main control loop.
    """

    logger.debug("Starting thread.")

    # Enter Control/Worker loops.
    if RANK == ROOT_RANK:
        id_distribution_loop(argc, cfg)
        events = None
    else:
        events = event_location_loop(argc, cfg)

    # Gather events from all threads.
    events = COMM.gather(events, root=ROOT_RANK)

    # Write event locations.
    if RANK == ROOT_RANK:
        events = pd.concat(events, ignore_index=True)
        write_events(events, argc.output_file)

    logger.info("Thread terminated without error.")

    return (True)


def configure_logging(verbose, logfile):
    """
    A utility function to configure logging.
    """

    if verbose is True:
        level = logging.DEBUG
    else:
        level = logging.INFO
    for name in (__name__,):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if level == logging.DEBUG:
            formatter = logging.Formatter(
                fmt="%(asctime)s::%(levelname)s::%(funcName)s()::%(lineno)d::"\
                    "{:s}::{:04d}:: %(message)s".format(PROCESSOR_NAME, RANK),
                datefmt="%Y%jT%H:%M:%S"
            )
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s::%(levelname)s::{:s}::{:04d}:: "\
                    "%(message)s".format(PROCESSOR_NAME, RANK),
                datefmt="%Y%jT%H:%M:%S"
            )
        if logfile is not None:
            file_handler = logging.FileHandler(logfile)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return (True)


def log_errors(func):
    """
    A decorator to for error logging.
    """

    def wrapped_func(*args, **kwargs):
        try:
            return (func(*args, **kwargs))
        except Exception as exc:
            logger.error(
                f"{func.__name__}() raised {type(exc)}: {exc}"
            )
            raise (exc)

    return (wrapped_func)


@log_errors
def signal_handler(sig, frame):
    """
    A utility function to to handle interrupting signals.
    """

    raise (SystemError("Interrupting signal received... aborting"))


@log_errors
def event_location_loop(argc, cfg):
    """
    A worker loop to locate events.
    """

    # Load network geometry from disk.
    stations = load_stations(argc.network_geometry)

    with pykonal.locate.EQLocator(
        station_dict(stations),
        cfg["traveltime_inventory"]
    ) as locator:

        # Load the input catalog.
        catalog = load_catalog(argc.catalog)
        arrivals = catalog["arrivals"]
        events = catalog["events"]
        events = events.set_index("event_id")

        # Initialize an empty dataframe to hold event locations.
        relocated_events = pd.DataFrame()

        # Give aliases for a few configuration parameters.
        dlat = cfg["dlat"]
        dlon = cfg["dlon"]
        dz   = cfg["dz"]
        dt   = cfg["dt"]

        dtheta = np.radians(dlat)
        dphi = np.radians(dlon)

        # Initialize the search region.
        delta = np.array([dz, dtheta, dphi, dt])

        while True:

            # Request an event
            COMM.send(RANK, dest=ROOT_RANK, tag=ID_REQUEST_TAG)
            event_id = COMM.recv(source=ROOT_RANK, tag=ID_TRANSMISSION_TAG)

            if event_id is None:
                logger.debug("Received sentinel.")
                return (relocated_events)

            logger.debug(f"Received ID #{event_id}.")

            # Extract the initial event location and convert to
            # spherical coordinates.
            _columns = ["latitude", "longitude", "depth", "time"]
            initial = events.loc[event_id, _columns].values
            initial[:3] = geo2sph(initial[:3])

            # Clear arrivals from previous event.
            locator.clear_arrivals()
            _arrival_dict = arrival_dict(arrivals, event_id)
            locator.add_arrivals(_arrival_dict)

            loc = locator.locate(initial, delta)

            # Get residual RMS, reformat result, and append to events
            # DataFrame.
            rms = locator.rms(loc)
            loc[:3] = sph2geo(loc[:3])
            event = pd.DataFrame(
                [np.concatenate((loc, [rms, event_id]))],
                columns=EVENT_FIELDS_OUTPUT
            )
            relocated_events = relocated_events.append(event, ignore_index=True)

    return (False)


@log_errors
def id_distribution_loop(argc, cfg):
    """
    A loop to distribute event IDs to hungry workers.
    """

    catalog = load_catalog(argc.catalog)
    events = catalog["events"]
    del (catalog)
    event_ids = events["event_id"].unique()[:50]
    # Distribute event IDs.
    for idx in range(len(event_ids)):
        event_id = event_ids[idx]
        requesting_rank = COMM.recv(source=MPI.ANY_SOURCE, tag=ID_REQUEST_TAG)
        COMM.send(event_id, dest=requesting_rank, tag=ID_TRANSMISSION_TAG)
    # Distribute sentinel.
    for irank in range(WORLD_SIZE - 1):
        requesting_rank = COMM.recv(source=MPI.ANY_SOURCE, tag=ID_REQUEST_TAG)
        COMM.send(None, dest=requesting_rank, tag=ID_TRANSMISSION_TAG)

    return (True)


@log_errors
def load_catalog(input_file):
    """
    Load and return arrival data from input file.

    Returns: pandas.DataFrame object with "network", "station", "phase",
    "time", and "event_id" fields.
    """
    input_file = pathlib.Path(input_file)
    if input_file.suffix in (".h5", ".hdf5"):
        return (_load_catalog_hdf5(input_file))
    else:
        raise (NotImplementedError())


@log_errors
def _load_catalog_hdf5(input_file):
    arrivals = pd.read_hdf(input_file, key="arrivals")
    events = pd.read_hdf(input_file, key="events")
    arrivals = arrivals[ARRIVAL_FIELDS_INPUT]
    events = events[EVENT_FIELDS_INPUT]
    catalog = dict(arrivals=arrivals, events=events)

    return (catalog)


@log_errors
def load_stations(input_file):
    """
    Load and return network geometry from input file.

    Input file must be HDF5 file created using pandas.HDFStore with a
    "stations" table that contains "network", "station", "latitude",
    "longitude", and "elevation" fields. Units of degrees are assumed
    for "latitude" and "longitude", and units of kilometers are assumed
    for "elevation".

    Returns: pandas.DataFrame object with "network", "station",
    "latitude", "longitude", and "depth" fields. Units of "depth" are
    kilometers.
    """

    input_file = pathlib.Path(input_file)
    if input_file.suffix in (".h5", ".hdf5"):
        return (_load_stations_hdf5(input_file))
    elif input_file.suffix in (".txt", ".dat"):
        return (_load_stations_txt(input_file))

@log_errors
def _load_stations_hdf5(input_file):
    with pd.HDFStore(input_file, mode="r") as store:
        stations = store["stations"]

    stations = stations[STATION_FIELDS_INPUT]
    stations["depth"] = -stations["elevation"]

    return (stations)


@log_errors
def _load_stations_txt(input_file):
    stations = pd.read_csv(input_file, delim_whitespace=True)

    stations = stations[STATION_FIELDS_INPUT]
    stations["depth"] = -stations["elevation"]

    return (stations)


@log_errors
def arrival_dict(dataframe, event_id):
    """
    Return a dictionary with phase-arrival data suitable for passing to
    the EQLocator.add_arrivals() method.

    Returned dictionary has ("station_id", "phase") keys, where
    "station_id" = f"{network}.{station}", and values are
    phase-arrival timestamps.
    """

    dataframe = dataframe.set_index("event_id")
    fields = ["network", "station", "phase", "time"]
    dataframe = dataframe.loc[event_id, fields]

    _arrival_dict = {
        (network, station, phase): timestamp
        for network, station, phase, timestamp in dataframe.values
    }

    return (_arrival_dict)


@log_errors
def station_dict(dataframe):
    """
    Return a dictionary with network geometry suitable for passing to
    the EQLocator constructor.

    Returned dictionary has "station_id" keys, where "station_id" =
    f"{network}.{station}", and values are spherical coordinates of
    station locations.
    """

    if np.any(dataframe[["network", "station"]].duplicated()):
        raise (IOError("Multiple coordinates supplied for single station(s)"))

    dataframe = dataframe.set_index(["network", "station"])

    _station_dict = {
        (network, station): geo2sph(
            dataframe.loc[
                (network, station),
                ["latitude", "longitude", "depth"]
            ].values
        ) for network, station in dataframe.index
    }

    return (_station_dict)


@log_errors
def write_events(dataframe, output_file):
    """
    Write event locations to HDF5 file via pandas.HDFStore.
    """

    logger.debug("Saving event locations to disk.")

    # Convert dtypes before saving event locations.
    for field in DTYPES:
        dataframe[field] = dataframe[field].astype(DTYPES[field])

    with pd.HDFStore(output_file, mode="w") as store:
        store["events"] = dataframe

    return (True)



if __name__ == "__main__":
    # Add some signal handlers to abort all threads.
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGCONT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Parse command line arguments.
    argc = parse_args()

    # Load parameters for parameter file.
    cfg = parse_cfg(argc.configuration_file)

    # Configure logging.
    configure_logging(argc.verbose, argc.log_file)
    logger = logging.getLogger(__name__)

    # Start the  main loop.
    main(argc, cfg)
