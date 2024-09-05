from enum import Enum

from skrl.utils.model_instantiators.jax.categorical import categorical_model
from skrl.utils.model_instantiators.jax.deterministic import deterministic_model
from skrl.utils.model_instantiators.jax.gaussian import gaussian_model


# keep for compatibility with versions prior to 1.3.0
class Shape(Enum):
    """
    Enum to select the shape of the model's inputs and outputs
    """
    ONE = 1
    STATES = 0
    OBSERVATIONS = 0
    ACTIONS = -1
    STATES_ACTIONS = -2
