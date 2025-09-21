from skrl.agents.jax.rpo.rpo import RPO
from skrl.agents.jax.rpo.rpo_cfg import RPO_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
RPO_DEFAULT_CONFIG = asdict(RPO_CFG())  # back compatibility config
# fmt: on
