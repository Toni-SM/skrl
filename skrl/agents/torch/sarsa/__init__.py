from skrl.agents.torch.sarsa.sarsa import SARSA
from skrl.agents.torch.sarsa.sarsa_cfg import SARSA_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
SARSA_DEFAULT_CONFIG = asdict(SARSA_CFG())  # back compatibility config
# fmt: on
