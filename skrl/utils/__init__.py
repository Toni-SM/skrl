from typing import Optional

import os
import random
import sys
import time

import numpy as np

from skrl import config, logger


def set_seed(seed: Optional[int] = None, deterministic: bool = False) -> int:
    """
    Set the seed for the random number generators

    Due to NumPy's legacy seeding constraint the seed must be between 0 and 2**32 - 1.
    Otherwise a NumPy exception (``ValueError: Seed must be between 0 and 2**32 - 1``) will be raised

    Modified packages:

    - random
    - numpy
    - torch (if available)
    - jax (skrl's PRNG key: ``config.jax.key``)

    Example::

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

    :param seed: The seed to set. Is None, a random seed will be generated (default: ``None``)
    :type seed: int, optional
    :param deterministic: Whether PyTorch is configured to use deterministic algorithms (default: ``False``).
                          The following environment variables should be established for CUDA 10.1 (``CUDA_LAUNCH_BLOCKING=1``)
                          and for CUDA 10.2 or later (``CUBLAS_WORKSPACE_CONFIG=:16:8`` or ``CUBLAS_WORKSPACE_CONFIG=:4096:8``).
                          See PyTorch `Reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`_ for details
    :type deterministic: bool, optional

    :return: Seed
    :rtype: int
    """
    # generate a random seed
    if seed is None:
        try:
            seed = int.from_bytes(os.urandom(4), byteorder=sys.byteorder)
        except NotImplementedError:
            seed = int(time.time() * 1000)
        seed %= 2 ** 31  # NumPy's legacy seeding seed must be between 0 and 2**32 - 1

    seed = int(seed)
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
