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

geo2sph = pykonal.transformations.geo2sph
sph2geo = pykonal.transformations.sph2geo

def parse_args():
    """
    Parse and return command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "arrivals_file",
        type=str,
        help="Input file containing phase arrival data."
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
    cfg["traveltime_directory"] = parser.get(
        "DEFAULT",
        "traveltime_directory",
        fallback=None
    )
    cfg["overwrite_traveltimes"] = parser.getboolean(
        "DEFAULT",
        "overwrite_traveltimes",
        fallback=False
    )
    cfg["pwave_velocity_model"] = parser.get(
        "DEFAULT",
        "pwave_velocity_model",
        fallback=None
    )
    cfg["swave_velocity_model"] = parser.get(
        "DEFAULT",
        "swave_velocity_model",
        fallback=None
    )
    cfg["dlat_search"] = parser.getfloat(
        "Differential Evolution",
        "dlat_search",
        fallback=0.1
    )
    cfg["dlon_search"] = parser.getfloat(
        "Differential Evolution",
        "dlon_search",
        fallback=0.1
    )
    cfg["dz_search"] = parser.getfloat(
        "Differential Evolution",
        "dz_search",
        fallback=10
    )
    cfg["dt_search"] = parser.getfloat(
        "Differential Evolution",
        "dt_search",
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

    # Define columns to output.
    columns = ["latitude", "longitude", "depth", "time", "residual", "event_id"]

    # Load network geometry from disk.
    stations = load_stations(argc.network_geometry)

    # Load velocity models.
    pwave_velocity = pykonal.fields.load(cfg["pwave_velocity_model"])
    swave_velocity = pykonal.fields.load(cfg["swave_velocity_model"])

    # Initialize EQLocator object.
    locator = pykonal.locate.EQLocator(
        station_dict(stations),
        tt_dir=cfg["traveltime_directory"]
    )
    locator.grid.min_coords     = pwave_velocity.min_coords
    locator.grid.node_intervals = pwave_velocity.node_intervals
    locator.grid.npts           = pwave_velocity.npts
    locator.pwave_velocity      = pwave_velocity.values
    locator.swave_velocity      = swave_velocity.values

    # Load arrival data from disk.
    arrivals = load_arrivals(argc.arrivals_file)

    # Initialize an empty dataframe to hold event locations.
    events = pd.DataFrame()

    # Give aliases for a few configuration parameters.
    dlat = cfg["dlat_search"]
    dlon = cfg["dlon_search"]
    dz   = cfg["dz_search"]
    dt   = cfg["dt_search"]

    while True:

        # Request an event
        COMM.send(RANK, dest=ROOT_RANK, tag=ID_REQUEST_TAG)
        event_id = COMM.recv(source=ROOT_RANK, tag=ID_TRANSMISSION_TAG)

        if event_id is None:
            logger.debug("Received sentinel.")
            return (events)

        logger.debug(f"Received ID #{event_id}.")

        # Clear arrivals from previous event.
        locator.clear_arrivals()
        locator.add_arrivals(arrival_dict(arrivals, event_id))
        locator.load_traveltimes()
        loc = locator.locate(dlat=dlat, dlon=dlon, dz=dz, dt=dt)

        # Get residual RMS, reformat result, and append to events
        # DataFrame.
        rms = locator.rms(loc)
        loc[:3] = sph2geo(loc[:3])
        event = pd.DataFrame(
            [np.concatenate((loc, [rms, event_id]))],
            columns=columns
        )
        events = events.append(event, ignore_index=True)

    return (False)


@log_errors
def id_distribution_loop(argc, cfg):
    """
    A loop to distribute event IDs to hungry workers.
    """

    arrivals = load_arrivals(argc.arrivals_file)
    event_ids = arrivals["event_id"].unique()
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
def load_arrivals(input_file):
    """
    Load and return arrival data from input file.

    Returns: pandas.DataFrame object with "network", "station", "phase",
    "time", and "event_id" fields.
    """

    with pd.HDFStore(input_file, mode="r") as store:
        arrivals = store["arrivals"]

    arrivals = arrivals[["network", "station", "phase", "time", "event_id"]]

    return (arrivals)


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

    with pd.HDFStore(input_file, mode="r") as store:
        stations = store["stations"]

    stations["depth"] = -stations["elevation"]
    stations = stations[
        ["network", "station", "latitude", "longitude", "depth"]
    ]

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
        (f"{network}.{station}", phase): timestamp
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
        f"{network}.{station}": geo2sph(
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
