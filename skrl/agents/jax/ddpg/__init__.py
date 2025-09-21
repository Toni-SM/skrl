from skrl.agents.jax.ddpg.ddpg import DDPG
from skrl.agents.jax.ddpg.ddpg_cfg import DDPG_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
DDPG_DEFAULT_CONFIG = asdict(DDPG_CFG())  # back compatibility config
# fmt: on
