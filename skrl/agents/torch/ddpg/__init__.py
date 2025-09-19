from skrl.agents.torch.ddpg.ddpg import DDPG
from skrl.agents.torch.ddpg.ddpg_cfg import DDPG_CFG
from skrl.agents.torch.ddpg.ddpg_rnn import DDPG_RNN


# fmt: off
from dataclasses import asdict  # isort:skip
DDPG_DEFAULT_CONFIG = asdict(DDPG_CFG())  # back compatibility config
# fmt: on
