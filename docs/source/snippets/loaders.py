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
