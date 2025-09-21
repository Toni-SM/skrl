from skrl.agents.jax.cem.cem import CEM
from skrl.agents.jax.cem.cem_cfg import CEM_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
CEM_DEFAULT_CONFIG = asdict(CEM_CFG())  # back compatibility config
# fmt: on
