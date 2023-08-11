# since Isaac Gym (preview) environments are implemented on top of PyTorch, the loaders are the same

from skrl.envs.loaders.torch import (  # isort:skip
    load_isaacgym_env_preview2,
    load_isaacgym_env_preview3,
    load_isaacgym_env_preview4,
)
