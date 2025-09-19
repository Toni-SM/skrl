from skrl.agents.torch.ppo.ppo import PPO
from skrl.agents.torch.ppo.ppo_cfg import PPO_CFG
from skrl.agents.torch.ppo.ppo_rnn import PPO_RNN


# fmt: off
from dataclasses import asdict  # isort:skip
PPO_DEFAULT_CONFIG = asdict(PPO_CFG())  # back compatibility config
# fmt: on
