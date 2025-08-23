from typing import Optional

import os
import random
import sys
import time

import numpy as np

from skrl import config, logger


def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    """Set the seed for the random number generators.

    .. note::

        In distributed runs, the worker/process seed will be incremented (counting from the defined value)
        according to its rank.

    .. warning::

        Due to NumPy's legacy seeding constraint the seed must be between 0 and 2**32 - 1.
        Otherwise a NumPy exception (``ValueError: Seed must be between 0 and 2**32 - 1``) will be raised.

    Modified packages:

    - ``random``
    - ``numpy``
    - ``torch`` (if available)
    - ``jax`` (skrl's PRNG key: ``config.jax.key``)

    Example:

    .. code-block:: python

        # fixed seed
        >>> from skrl.utils import set_seed
        >>> set_seed(42)
        [skrl:INFO] Seed: 42
        42

        # random seed
        >>> from skrl.utils import set_seed
        >>> set_seed()
        [skrl:INFO] Seed: 1776118066
        1776118066

        # enable deterministic. The following environment variables should be established:
        # - CUDA 10.1: CUDA_LAUNCH_BLOCKING=1
        # - CUDA 10.2 or later: CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:8
        >>> from skrl.utils import set_seed
        >>> set_seed(42, deterministic=True)
        [skrl:INFO] Seed: 42
        [skrl:WARNING] PyTorch/cuDNN deterministic algorithms are enabled. This may affect performance
        42

    :param seed: The seed to set. If ``None``, a random seed will be generated.
    :param deterministic: Whether PyTorch is configured to use deterministic algorithms.
        The following environment variables should be established for CUDA 10.1 (``CUDA_LAUNCH_BLOCKING=1``)
        and for CUDA 10.2 or later (``CUBLAS_WORKSPACE_CONFIG=:16:8`` or ``CUBLAS_WORKSPACE_CONFIG=:4096:8``).
        See PyTorch `Reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`_ for details.

    :return: Seed.
    """
    # generate a random seed
    if seed is None:
        try:
            seed = int.from_bytes(os.urandom(4), byteorder=sys.byteorder)
        except NotImplementedError:
            seed = int(time.time() * 1000)
        seed %= 2**31  # NumPy's legacy seeding seed must be between 0 and 2**32 - 1
    seed = int(seed)

    # set different seeds in distributed runs
    if config.torch.is_distributed:
        seed += config.torch.rank
    if config.jax.is_distributed:
        seed += config.jax.rank

    logger.info(f"Seed: {seed}")

    # numpy
    random.seed(seed)
    np.random.seed(seed)

    # torch
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

            # On CUDA 10.1, set environment variable CUDA_LAUNCH_BLOCKING=1
            # On CUDA 10.2 or later, set environment variable CUBLAS_WORKSPACE_CONFIG=:16:8 or CUBLAS_WORKSPACE_CONFIG=:4096:8

            logger.warning("PyTorch/cuDNN deterministic algorithms are enabled. This may affect performance")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"PyTorch seeding error: {e}")

    # jax
    config.jax.key = seed

    return seed


class ScopedTimer:
    """Scoped timer that can be used to time the execution of a block of code."""

    def __enter__(self):
        self._elapsed_time = None
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._elapsed_time = time.time() - self._start_time

    @property
    def elapsed_time(self) -> float:
        """Elapsed time (in seconds).

        .. note::

            If called within the scope of the context manager, the elapsed time is updated to reflect the time
            spent within the scope. If called outside the context manager scope, the elapsed time is fixed to
            the time at which the context manager was exited.

        :return: Elapsed time in seconds.
        """
        if self._elapsed_time is None:
            return time.time() - self._start_time
        return self._elapsed_time

    @property
    def elapsed_time_ms(self) -> float:
        """Elapsed time (in milliseconds).

        .. note::

            If called within the scope of the context manager, the elapsed time is updated to reflect the time
            spent within the scope. If called outside the context manager scope, the elapsed time is fixed to
            the time at which the context manager was exited.

        :return: Elapsed time in milliseconds.
        """
        if self._elapsed_time is None:
            return (time.time() - self._start_time) * 1000
        return self._elapsed_time * 1000
