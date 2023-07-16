from typing import Union

import logging
import sys

import numpy as np


__all__ = ["__version__", "logger", "config"]


# read library version from metadata
try:
    import importlib.metadata
    __version__ = importlib.metadata.version("skrl")
except ImportError:
    __version__ = "unknown"


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


# machine learning framework configuration
class _Config(object):
    def __init__(self) -> None:
        """Machine learning framework specific configuration
        """
        class JAX(object):
            def __init__(self) -> None:
                """JAX configuration
                """
                self._backend = "numpy"
                self._key = np.array([0, 0], dtype=np.uint32)

            @property
            def backend(self) -> str:
                """Backend used by the different components to operate and generate arrays

                This configuration excludes models and optimizers.
                Supported backend are: ``"numpy"`` and ``"jax"``
                """
                return self._backend

            @backend.setter
            def backend(self, value: str) -> None:
                if value not in ["numpy", "jax"]:
                    raise ValueError("Invalid jax backend. Supported values are: numpy, jax")
                self._backend = value

            @property
            def key(self) -> "jax.Array":
                """Pseudo-random number generator (PRNG) key
                """
                if isinstance(self._key, np.ndarray):
                    try:
                        import jax
                        self._key = jax.random.PRNGKey(self._key[1])
                    except ImportError:
                        pass
                return self._key

            @key.setter
            def key(self, value: Union[int, "jax.Array"]) -> None:
                if type(value) is int:
                    # don't import JAX if it has not been imported before
                    if "jax" in sys.modules:
                        import jax
                        value = jax.random.PRNGKey(value)
                    else:
                        value = np.array([0, value], dtype=np.uint32)
                self._key = value

        self.jax = JAX()

config = _Config()
