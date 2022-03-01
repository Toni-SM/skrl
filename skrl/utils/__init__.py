import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Sets the seed for all random number generators

    :param seed: The seed for all random number generators
    :type seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    