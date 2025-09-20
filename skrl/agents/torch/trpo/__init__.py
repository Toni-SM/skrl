from skrl.agents.torch.trpo.trpo import TRPO
from skrl.agents.torch.trpo.trpo_cfg import TRPO_CFG
from skrl.agents.torch.trpo.trpo_rnn import TRPO_RNN


# fmt: off
from dataclasses import asdict  # isort:skip
TRPO_DEFAULT_CONFIG = asdict(TRPO_CFG())  # back compatibility config
# fmt: on
