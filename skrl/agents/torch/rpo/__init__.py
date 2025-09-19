from skrl.agents.torch.rpo.rpo import RPO
from skrl.agents.torch.rpo.rpo_cfg import RPO_CFG
from skrl.agents.torch.rpo.rpo_rnn import RPO_RNN


# fmt: off
from dataclasses import asdict  # isort:skip
RPO_DEFAULT_CONFIG = asdict(RPO_CFG())  # back compatibility config
# fmt: on
