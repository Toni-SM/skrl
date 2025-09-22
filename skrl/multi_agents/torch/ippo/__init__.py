from skrl.multi_agents.torch.ippo.ippo import IPPO
from skrl.multi_agents.torch.ippo.ippo_cfg import IPPO_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
IPPO_DEFAULT_CONFIG = asdict(IPPO_CFG())  # back compatibility config
# fmt: on
