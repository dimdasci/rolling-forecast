"""
A collection of functions for common file I/O and logging tasks.

This module provides a collection of functions for common file I/O and
logging tasks. These functions can be used to load and save Python objects
to binary files, get the absolute path to a file in a project directory,
and set up logging handlers and formats.

Functions:
- load_params(): Loads parameters from the params.yaml file in the
  project root directory.
- setup_logging(): Sets up logging handlers and formats for a logger.
- get_project_dir(): Returns the absolute path to the project root directory.
- get_abs_path(rel_path: str, filename: str) -> str: Returns the absolute
  path to a file with a given relative path and filename.
- load_pickle(path: str) -> Any: Loads a Python object from a binary file
  using the pickle module.
- save_pickle(obj: Any, path: str) -> None: Saves a Python object to a binary
  file using the pickle module.

Usage:
    To use this module, simply import it and call the desired functions.

    Example:

        from src.utils import functions

        # Load parameters from params.yaml file
        params = functions.load_params()

        # Set up logging
        logger = functions.setup_logging(logname="my_logger",
                                         logfile="logs/app.log",
                                         loglevel="INFO")

        # Get project directory
        project_dir = functions.get_project_dir()

        # Get absolute path to a file
        abs_path = functions.get_abs_path(rel_path="data",
                                          filename="my_data.csv")

        # Load a Python object from a binary file
        my_obj = functions.load_pickle(path="my_obj.pkl")

        # Save a Python object to a binary file
        functions.save_pickle(obj=my_obj, path="new_obj.pkl")
"""


import yaml
import logging
from pathlib import Path
import os
import pickle
from typing import Any


def load_params() -> dict:
    """
    Load parameters from params.yaml in the project root directory.

    Returns:
        A dictionary containing the loaded parameters.
    """
    params_path = os.path.join(get_project_dir(), "params.yaml")
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    return params


def setup_logging(
    logname: str = "", logfile: str = "logs/app.log", loglevel: str = "DEBUG"
) -> logging.Logger:
    """
    Set up a logger with specified name, file path and log level, and return
    the configured logger.

    Params:
        logname: str, optional
            The name of the logger. If not provided, the root logger is used.
        logfile: str, optional
            The path to the log file. Default is 'logs/app.log'.
        loglevel: str, optional
            The log level to use. Default is 'DEBUG'.

    Returns:
        logging.Logger
            The configured logger object.

    The function sets up a logger with a specified name and log level, and adds
    two handlers: a file handler and a stream handler. The file handler writes
    log messages to the specified file, and the stream handler writes log
    messages to the console. The log messages include the current time, log
    level, filename, function name, line number, and log message.
    """

    loglevel = getattr(logging, loglevel)

    logger = logging.getLogger(logname)
    logger.setLevel(loglevel)
    fmt = (
        "%(asctime)s: %(levelname)s: %(filename)s: "
        + "%(funcName)s(): %(lineno)d: %(message)s"
    )
    formatter = logging.Formatter(fmt)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setLevel(loglevel)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def get_project_dir() -> str:
    """
    Returns the path to the project directory.

    Returns:
        str
            The path to the project directory.

    This function assumes that the current file is located within the project
    directory. It returns the path to the project  directory by resolving
    the absolute path of the current file, and then accessing its third parent
    directory.
    """
    return Path(__file__).resolve().parents[2]


def get_abs_path(rel_path: str, filename: str) -> str:
    """
    Returns the absolute path to a file given its relative path and filename.

    Params:
        rel_path: str
            The relative path to the file.
        filename: str
            The name of the file.

    Returns:
        str
            The absolute path to the file.

    This function returns the absolute path to a file given its relative path
    and filename. It first gets the path to the project directory by calling
    the `get_project_dir()` function, and then uses the `os.path.join()`
    method to join the project directory path, the relative path, and the
    filename together. The resulting path is returned as a string.
    """

    return os.path.join(
        get_project_dir(),
        rel_path,
        filename,
    )


def load_pickle(path: str) -> Any:
    """
    Loads a Python object from a binary file using the pickle module.

    Args:
        path (str): The path to the binary file.

    Returns:
        Any: The Python object loaded from the binary file.

    This function uses the Python pickle module to load a Python object from
    a binary file located at the specified path. The binary file should have
    been created using the pickle.dump() method. The loaded object is returned
    as the function's  output. If the file cannot be opened or the object
    cannot be loaded, a Python exception will be raised.
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def save_pickle(obj: Any, path: str) -> None:
    """
    Saves a Python object to a binary file using the pickle module.

    Args:
        obj (Any): The Python object to be saved.
        path (str): The path to the binary file.

    Returns:
        None: This function does not return any value.

    This function uses the Python pickle module to save a Python object to
    a binary file located at the specified path. The saved object can later
    be loaded using the load_pickle() function. If the file cannot be
    created or the object  cannot be saved, a Python exception will be raised.
    """

    with open(path, "wb") as f:
        pickle.dump(obj, f)
