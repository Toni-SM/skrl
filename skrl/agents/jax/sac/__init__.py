from skrl.agents.jax.sac.sac import SAC
from skrl.agents.jax.sac.sac_cfg import SAC_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
SAC_DEFAULT_CONFIG = asdict(SAC_CFG())  # back compatibility config
# fmt: on
