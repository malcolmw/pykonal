"""
An MPI script to locate earthquakes.

.. author: Malcolm C. A. White
"""

import argparse
import configparser
import emcee
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd
import pathlib
import pykonal
import scipy.special
import signal
import warnings

warnings.filterwarnings("ignore")

DTYPES = {
    "latitude":      np.float32,
    "longitude":     np.float32,
    "depth":         np.float32,
    "time":          np.float64,
    "residual":      np.float32,
    "event_id":      np.int32,
    "latitude_mad":  np.float32,
    "longitude_mad": np.float32,
    "depth_mad":     np.float32,
    "time_mad":      np.float64,
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
EVENT_FIELDS_OUTPUT = [
    *EVENT_FIELDS_INPUT,
    "residual",
    "latitude_mad",
    "longitude_mad",
    "depth_mad",
    "time_mad",
]
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
        "pick_errors",
        type=str,
        help="Input file containing pick errors."
    )
    parser.add_argument(
        "traveltime_inventory",
        type=str,
        help="Input traveltime inventory."
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
        "-e",
        "--enddate",
        type=str,
        help="End date."
    )
    parser.add_argument(
        "-n",
        "--nproc",
        type=int,
        default=mp.cpu_count(),
        help="Number of processors to use."
    )
    parser.add_argument(
        "-s",
        "--startdate",
        type=str,
        help="Start date."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Be verbose."
    )

    argc = parser.parse_args()

    if argc.startdate is not None:
        argc.startdate = pd.to_datetime(argc.startdate, format="%Y%j")
    if argc.enddate is not None:
        argc.enddate = pd.to_datetime(argc.enddate, format="%Y%j")

    return (argc)


def parse_cfg(configuration_file):
    """
    Parse and return contents of the configuration file.
    """

    cfg = dict()
    parser = configparser.ConfigParser()
    parser.read(configuration_file)

    cfg["mu_P"] = parser.getfloat(
        "DEFAULT",
        "mu_P"
    )
    cfg["sigma_P"] = parser.getfloat(
        "DEFAULT",
        "sigma_P"
    )
    cfg["mu_S"] = parser.getfloat(
        "DEFAULT",
        "mu_S"
    )
    cfg["sigma_S"] = parser.getfloat(
        "DEFAULT",
        "sigma_S"
    )
    cfg["confidence_level"] = parser.getfloat(
        "DEFAULT",
        "confidence_level",
        fallback=90.
    )
    cfg["nwalkers"] = parser.getint(
        "DEFAULT",
        "nwalkers",
        fallback=32
    )
    cfg["nsamples"] = parser.getint(
        "DEFAULT",
        "nsamples",
        fallback=1024
    )
    cfg["max_samples"] = parser.getint(
        "DEFAULT",
        "max_samples",
        fallback=32768
    )
    for key in cfg:
        if cfg[key] == "None":
            cfg[key] = None

    return (cfg)


def main(argc, cfg):
    """
    This is the main control loop.
    """

    logger.debug("Starting main().")

    # Load catalog from disk.
    catalog = load_catalog(
        argc.catalog,
        starttime=argc.startdate,
        endtime=argc.enddate
    )

    # Load network geometry from disk.
    stations = load_stations(argc.network_geometry)

    # Load pick errors from disk.
    pick_errors = load_pick_errors(argc.pick_errors)

    # Locate events
    events = locate_events(catalog, stations, pick_errors, argc, cfg)

    # Write event locations.
    write_events(events, argc.output_file)

    logger.info("Finished relocating events.")

    return (True)


class VoigtProfile(object):
    def __init__(self, mu=0, sigma=1, gamma=1):
        self.mu    = mu
        self.sigma = sigma
        self.gamma = gamma

    def logpdf(self, x):
        return (np.log(scipy.special.voigt_profile(x - self.mu, self.sigma, self.gamma)))


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
                    " %(message)s",
                datefmt="%Y%jT%H:%M:%S"
            )
        else:
            formatter = logging.Formatter(
                fmt="%(asctime)s::%(levelname)s::%(message)s",
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


def log_probability(args):
    global gl_LOCATOR

    rho, theta, phi, time = args

    log_likelihood = gl_LOCATOR.log_likelihood(args)

    if np.isnan(log_likelihood):
        return (-np.inf)
    else:
        log_prior = np.log(rho**2 + np.sin(theta))
        return (log_likelihood + log_prior)


@log_errors
def locate_events(catalog, stations, pick_errors, argc, cfg):
    """
    A worker loop to locate events.
    """

    global gl_LOCATOR

    pick_errors = pick_errors.set_index(["network", "station", "phase"])

    residual_rvs = dict()
    for network, station, phase in pick_errors.index:
        mu, gamma = pick_errors.loc[(network, station, phase), ["loc", "scale"]]
        mu = (cfg[f"mu_{phase}"] + mu) / 2
        residual_rvs[(network, station, phase)] = VoigtProfile(
            mu=mu,
            sigma=cfg[f"sigma_{phase}"],
            gamma=gamma
        )

    # Load network geometry from disk.
    stations = load_stations(argc.network_geometry)

    with pykonal.locate.EQLocator(
        station_dict(stations),
        argc.traveltime_inventory
    ) as gl_LOCATOR:

        arrivals = catalog["arrivals"]
        events = catalog["events"]
        events = events.set_index("event_id")

        # Initialize an empty list to hold updated event locations.
        relocated_events = list()

        for event_id, event in events.iterrows():

            logger.info(f"Relocating event ID #{event_id}.")

            # Clear arrivals from previous event.
            gl_LOCATOR.clear_arrivals()
            gl_LOCATOR.add_arrivals(arrival_dict(arrivals, event_id))

            gl_LOCATOR.residual_rvs = {
                (network, station, phase):
                    residual_rvs[(network, station, phase)]
                    if (network, station, phase) in residual_rvs
                    else residual_rvs[("XX", "XX", phase)]
                for network, station, phase in gl_LOCATOR.arrivals
            }

            model0 = geo2sph(event[["latitude", "longitude", "depth"]])
            model0 = np.append(model0, event["time"])
            pos = model0 + 1e-4 * np.random.randn(cfg["nwalkers"], 4)
            nwalkers, ndim = pos.shape

            gl_LOCATOR.read_traveltimes()

            with mp.Pool(argc.nproc) as pool:
                nsamples = cfg["nsamples"]
                while True:
                    sampler = emcee.EnsembleSampler(
                        nwalkers,
                        ndim,
                        log_probability,
                        pool=pool
                    )
                    sampler.run_mcmc(pos, nsamples, progress=True)
                    try:
                        discard = int(5 * np.mean(sampler.get_autocorr_time()))
                    except (emcee.autocorr.AutocorrError, ValueError) as exc:
                        nsamples *= 2
                        if nsamples > cfg["max_samples"]:
                            success = False
                            break
                        continue
                    success = True
                    break

            if success is False:
                logger.info(f"Failed to relocate event ID #{event_id}.")
                continue

            log_prob = sampler.get_log_prob()
            chain    = sampler.get_chain().copy()

            loc = chain[np.unravel_index(np.argmax(log_prob), log_prob.shape)]
            rms = gl_LOCATOR.rms(loc)
            loc[:3] = sph2geo(loc[:3])

            flat_samples = sampler.get_chain(discard=discard, flat=True).copy()
            flat_samples[:, :3] = sph2geo(flat_samples[:, :3])

            mad = np.mean(np.abs(flat_samples - loc), axis=0)

            for i, label in enumerate(("latitude", "longitude", "depth", "time")):
                logger.debug(f"{label} = {loc[i]:.3f} +/- {mad[i]:.3f}")

            relocated_events.append(
                np.concatenate(
                    (loc, [event_id, rms], mad)
                )
            )

    events = pd.DataFrame(relocated_events, columns=EVENT_FIELDS_OUTPUT)

    return (events)


@log_errors
def load_catalog(input_file, starttime=None, endtime=None):
    """
    Load and return arrival data from input file.

    Returns: pandas.DataFrame object with "network", "station", "phase",
    "time", and "event_id" fields.
    """
    input_file = pathlib.Path(input_file)
    if input_file.suffix in (".h5", ".hdf5"):
        catalog = _load_catalog_hdf5(input_file)
    else:
        raise (NotImplementedError())

    # Subset catalog for the time range of interest.
    if starttime is not None:
        catalog["events"] = catalog["events"][
            pd.to_datetime(catalog["events"]["time"]*1e9) > starttime
        ]
    if endtime is not None:
        catalog["events"] = catalog["events"][
            pd.to_datetime(catalog["events"]["time"]*1e9) < endtime
        ]
    catalog["arrivals"] = catalog["arrivals"][
        catalog["arrivals"]["event_id"].isin(
            catalog["events"]["event_id"]
        )
    ]

    return (catalog)


@log_errors
def _load_catalog_hdf5(input_file):
    arrivals = pd.read_hdf(input_file, key="arrivals")
    events = pd.read_hdf(input_file, key="events")
    arrivals = arrivals[ARRIVAL_FIELDS_INPUT]
    events = events[EVENT_FIELDS_INPUT]
    catalog = dict(arrivals=arrivals, events=events)

    return (catalog)


@log_errors
def load_pick_errors(input_file):
    """
    Load and return pick errors from input file.
    """

    pick_errors = pd.read_hdf(input_file)

    return (pick_errors)


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

    stations = stations.drop_duplicates(["network", "station"])

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
    # Parse command line arguments.
    argc = parse_args()

    # Load parameters for parameter file.
    cfg = parse_cfg(argc.configuration_file)

    # Configure logging.
    configure_logging(argc.verbose, argc.log_file)
    logger = logging.getLogger(__name__)

    # Start the  main loop.
    main(argc, cfg)
