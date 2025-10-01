# [start-isaaclab-envs-parameters-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_isaaclab_env

# load environment
env = load_isaaclab_env(task_name="Isaac-Cartpole-v0")
# [end-isaaclab-envs-parameters-torch]


# [start-isaaclab-envs-parameters-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_isaaclab_env

# load environment
env = load_isaaclab_env(task_name="Isaac-Cartpole-v0")
# [end-isaaclab-envs-parameters-jax]


# [start-isaaclab-envs-cli-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_isaaclab_env

# load environment
env = load_isaaclab_env()
# [end-isaaclab-envs-cli-torch]


# [start-isaaclab-envs-cli-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_isaaclab_env

# load environment
env = load_isaaclab_env()
# [end-isaaclab-envs-cli-jax]

# =============================================================================

# [start-isaac-gym-envs-preview-4-api]
import isaacgymenvs

env = isaacgymenvs.make(seed=0,
                        task="Cartpole",
                        num_envs=2000,
                        sim_device="cuda:0",
                        rl_device="cuda:0",
                        graphics_device_id=0,
                        headless=False)
# [end-isaac-gym-envs-preview-4-api]


# [start-isaac-gym-envs-preview-4-parameters-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_isaacgym_env_preview4

# load environment
env = load_isaacgym_env_preview4(task_name="Cartpole")
# [end-isaac-gym-envs-preview-4-parameters-torch]


# [start-isaac-gym-envs-preview-4-parameters-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_isaacgym_env_preview4

# load environment
env = load_isaacgym_env_preview4(task_name="Cartpole")
# [end-isaac-gym-envs-preview-4-parameters-jax]


# [start-isaac-gym-envs-preview-4-cli-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_isaacgym_env_preview4

# load environment
env = load_isaacgym_env_preview4()
# [end-isaac-gym-envs-preview-4-cli-torch]


# [start-isaac-gym-envs-preview-4-cli-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_isaacgym_env_preview4

# load environment
env = load_isaacgym_env_preview4()
# [end-isaac-gym-envs-preview-4-cli-jax]
