import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set the seed for the random number generators

    Modified packages:

    - random
    - numpy
    - torch

    Example::

        >>> from skrl.utils import set_seed
        >>> set_seed(42)

    :param seed: The seed to set
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    