from skrl.agents.torch.amp.amp import AMP
from skrl.agents.torch.amp.amp_cfg import AMP_CFG


# fmt: off
from dataclasses import asdict  # isort:skip
AMP_DEFAULT_CONFIG = asdict(AMP_CFG())  # back compatibility config
# fmt: on
