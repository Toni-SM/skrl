from skrl.multi_agents.jax.mappo.mappo import MAPPO
from skrl.multi_agents.jax.mappo.mappo_cfg import MAPPO_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
MAPPO_DEFAULT_CONFIG = asdict(MAPPO_CFG())  # back compatibility config
# fmt: on
