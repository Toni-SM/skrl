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


# [start-isaaclab-envs-parameters-warp]
# import the environment loader
from skrl.envs.loaders.warp import load_isaaclab_env

# load environment
env = load_isaaclab_env(task_name="Isaac-Cartpole-v0")
# [end-isaaclab-envs-parameters-warp]


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


# [start-isaaclab-envs-cli-warp]
# import the environment loader
from skrl.envs.loaders.warp import load_isaaclab_env

# load environment
env = load_isaaclab_env()
# [end-isaaclab-envs-cli-warp]

# =============================================================================

# [start-playground-envs-parameters-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_playground_env

# load environment
env = load_playground_env(task_name="CartpoleBalance", num_envs=1024, episode_length=300)
# [end-playground-envs-parameters-torch]


# [start-playground-envs-parameters-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_playground_env

# load environment
env = load_playground_env(task_name="CartpoleBalance", num_envs=1024, episode_length=300)
# [end-playground-envs-parameters-jax]


# [start-playground-envs-parameters-warp]
# import the environment loader
from skrl.envs.loaders.warp import load_playground_env

# load environment
env = load_playground_env(task_name="CartpoleBalance", num_envs=1024, episode_length=300)
# [end-playground-envs-parameters-warp]


# [start-playground-envs-cli-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_playground_env

# load environment
env = load_playground_env()
# [end-playground-envs-cli-torch]


# [start-playground-envs-cli-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_playground_env

# load environment
env = load_playground_env()
# [end-playground-envs-cli-jax]


# [start-playground-envs-cli-warp]
# import the environment loader
from skrl.envs.loaders.warp import load_playground_env

# load environment
env = load_playground_env()
# [end-playground-envs-cli-warp]
