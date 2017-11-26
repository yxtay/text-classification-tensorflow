import logging
from logging.handlers import RotatingFileHandler
import os
from typing import Iterable

from utils import path_join


def get_logger(name: str, log_path: str = os.path.join(os.path.dirname(__file__), "main.log"),
               console: bool = False) -> logging.Logger:
    """
    Simple logging wrapper that returns logger
    configured to log into file and console.

    Args:
        name (str): name of logger
        log_path (str): path of log file
        console (bool): whether to log on console

    Returns:
        logging.Logger: configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # ensure that logging handlers are not duplicated
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    # rotating file handler
    if log_path:
        fh = RotatingFileHandler(path_join(log_path),
                                 maxBytes=10 * 2 ** 20,  # 10 MB
                                 backupCount=1)  # 1 backup
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # console handler
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # null handler
    if not (log_path or console):
        logger.addHandler(logging.NullHandler())

    return logger


def get_name(name, file: str) -> str:
    """
    get logger name as module or file name

    Args:
        name: __name__, module name
        file: __file__, file name

    Returns:
        logger name
    """
    return os.path.basename(file) if name == "__main__" else name


def float_array_string(arr: Iterable[float]) -> str:
    """
    format array of floats to 4 decimal places

    Args:
        arr: array of floats

    Returns:
        formatted string
    """
    return "[" + ", ".join(["{:.4f}".format(el) for el in arr]) + "]"
