import argparse
import logging
import mpi4py.MPI as MPI
import pandas as pd
import signal


PROCESSOR_NAME      = MPI.Get_processor_name()
COMM                = MPI.COMM_WORLD
WORLD_SIZE          = COMM.Get_size()
RANK                = COMM.Get_rank()
ROOT_RANK           = 0
ID_REQUEST_TAG      = 100
ID_TRANSMISSION_TAG = 101


def parse_args():
    """
    Parse and return command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=str,
        help="Input file."
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

    return (None)


def main(argc, cfg):
    """
    This is the main control loop.
    """
    logger.debug("Starting thread.")
    if RANK == ROOT_RANK:
        id_distribution_loop(argc, cfg)
    else:
        event_location_loop(argc, cfg)

    logger.info("Thread completed successfully.")

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
    # Load arrival data from disk.
    arrivals = load_arrivals(argc.input_file)

    while True:
        # Request an event
        COMM.send(RANK, dest=ROOT_RANK, tag=ID_REQUEST_TAG)
        event_id = COMM.recv(source=ROOT_RANK, tag=ID_TRANSMISSION_TAG)
        if event_id is None:
            logger.debug("Received sentinel.")
            return (True)
        logger.debug(f"Received ID #{event_id}.")



@log_errors
def id_distribution_loop(argc, cfg):
    """
    A loop to distribute event IDs to hungry workers.
    """
    arrivals = load_arrivals(argc.input_file)
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
