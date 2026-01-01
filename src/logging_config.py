"""Shared logging setup for the project.

Provides a helper to configure module loggers with a file handler
writing to `logs.log` in the project root. This keeps logging
configuration consistent across modules.
"""

import logging
import os
from typing import Optional


def configure_file_logger(name: Optional[str] = None, log_file: Optional[str] = None, level: int = logging.DEBUG) -> logging.Logger:
    """Return a logger configured with a FileHandler writing to `logs.log`.

    If a FileHandler for the same `log_file` is already attached to the
    logger, this function leaves it in place.
    """
    if log_file is None:
        log_file = os.path.join(os.getcwd(), "logs.log")

    logger = logging.getLogger(name)
    # Ensure callers can safely call this function multiple times
    logger.addHandler(logging.NullHandler())

    # If a file handler for this exact file already exists, do nothing
    if any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_file
           for h in logger.handlers):
        return logger

    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.setLevel(level)

        logger.addHandler(handler)
        logger.setLevel(level)
        logger.debug("Logging initialized to %s", log_file)
    except Exception:
        # Never let logging setup break the application
        pass

    return logger
