from skrl import logger


try:
    from warp_nn.activations import ELU, SELU, LeakyReLU, ReLU, Sigmoid, SoftPlus, SoftSign, Tanh
    from warp_nn.modules import LazyLinear, Linear, Module, Parameter, Sequential

    logger.info("Using external warp-nn package")
except (ImportError, ModuleNotFoundError) as e:
    from . import functional, init
    from .activations import ELU, ReLU, Tanh
    from .flatten import Flatten
    from .linear import LazyLinear, Linear
    from .module import Module
    from .parameter import Parameter
    from .sequential import Sequential

    logger.warning(f"External warp-nn package not found, using internal implementations: {e}")
