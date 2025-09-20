from skrl.agents.torch.td3.td3 import TD3
from skrl.agents.torch.td3.td3_cfg import TD3_CFG
from skrl.agents.torch.td3.td3_rnn import TD3_RNN


# fmt: off
from dataclasses import asdict  # isort:skip
TD3_DEFAULT_CONFIG = asdict(TD3_CFG())  # back compatibility config
# fmt: on
