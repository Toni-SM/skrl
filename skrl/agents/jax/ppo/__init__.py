from skrl.agents.jax.ppo.ppo import PPO
from skrl.agents.jax.ppo.ppo_cfg import PPO_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
PPO_DEFAULT_CONFIG = asdict(PPO_CFG())  # back compatibility config
# fmt: on
