# TODO: Delete this file in future releases

from skrl import logger  # isort: skip
logger.warning("Using `from skrl.envs.torch import ...` is deprecated and will be removed in future versions.")
logger.warning(" - Import loaders using `from skrl.envs.loaders.torch import ...`")
logger.warning(" - Import wrappers using `from skrl.envs.wrappers.torch import ...`")


from skrl.envs.loaders.torch import (
    load_bidexhands_env,
    load_isaacgym_env_preview2,
    load_isaacgym_env_preview3,
    load_isaacgym_env_preview4,
    load_isaaclab_env,
    load_omniverse_isaacgym_env
)
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper, wrap_env
