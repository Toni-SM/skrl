from skrl.agents.jax.td3.td3 import TD3
from skrl.agents.jax.td3.td3_cfg import TD3_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
TD3_DEFAULT_CONFIG = asdict(TD3_CFG())  # back compatibility config
# fmt: on
