import os
import logging

__all__ = ["__version__", "logger"]


# read library version from file
path = os.path.join(os.path.dirname(__file__), "version.txt")
with open(path, "r") as file:
    __version__ = file.read().strip()


# logger with format
class _Formatter(logging.Formatter):
    _format = "[%(name)s:%(levelname)s] %(message)s"
    _formats = {logging.DEBUG: f"\x1b[38;20m{_format}\x1b[0m",
                logging.INFO: f"\x1b[38;20m{_format}\x1b[0m",
                logging.WARNING: f"\x1b[33;20m{_format}\x1b[0m",
                logging.ERROR: f"\x1b[31;20m{_format}\x1b[0m",
                logging.CRITICAL: f"\x1b[31;1m{_format}\x1b[0m"}

    def format(self, record):
        return logging.Formatter(self._formats.get(record.levelno)).format(record)

_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_handler.setFormatter(_Formatter())

logger = logging.getLogger("skrl")
logger.setLevel(logging.DEBUG)
logger.addHandler(_handler)
