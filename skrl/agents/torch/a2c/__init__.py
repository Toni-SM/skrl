from skrl.agents.torch.a2c.a2c import A2C
from skrl.agents.torch.a2c.a2c_cfg import A2C_CFG
from skrl.agents.torch.a2c.a2c_rnn import A2C_RNN


# fmt: off
from dataclasses import asdict  # isort:skip
A2C_DEFAULT_CONFIG = asdict(A2C_CFG())  # back compatibility config
# fmt: on
