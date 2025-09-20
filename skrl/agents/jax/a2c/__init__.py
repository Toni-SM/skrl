from skrl.agents.jax.a2c.a2c import A2C
from skrl.agents.jax.a2c.a2c_cfg import A2C_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
A2C_DEFAULT_CONFIG = asdict(A2C_CFG())  # back compatibility config
# fmt: on
