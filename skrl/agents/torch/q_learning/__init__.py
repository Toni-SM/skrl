from skrl.agents.torch.q_learning.q_learning import Q_LEARNING
from skrl.agents.torch.q_learning.q_learning_cfg import Q_LEARNING_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
Q_LEARNING_DEFAULT_CONFIG = asdict(Q_LEARNING_CFG())  # back compatibility config
# fmt: on
