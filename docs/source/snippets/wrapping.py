# [pytorch-start-isaaclab]
# import the environment wrapper and loader
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.loaders.torch import load_isaaclab_env

# load the environment
env = load_isaaclab_env(task_name="Isaac-Cartpole-Direct-v0")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaaclab")'
# [pytorch-end-isaaclab]


# [jax-start-isaaclab]
# import the environment wrapper and loader
from skrl.envs.wrappers.jax import wrap_env
from skrl.envs.loaders.jax import load_isaaclab_env

# load the environment
env = load_isaaclab_env(task_name="Isaac-Cartpole-Direct-v0")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaaclab")'
# [jax-end-isaaclab]


# [pytorch-start-isaaclab-multi-agent]
# import the environment wrapper and loader
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.loaders.torch import load_isaaclab_env

# load the environment
env = load_isaaclab_env(task_name="Isaac-Cart-Double-Pendulum-Direct-v0")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaaclab-multi-agent")'
# [pytorch-end-isaaclab-multi-agent]


# [jax-start-isaaclab-multi-agent]
# import the environment wrapper and loader
from skrl.envs.wrappers.jax import wrap_env
from skrl.envs.loaders.jax import load_isaaclab_env

# load the environment
env = load_isaaclab_env(task_name="Isaac-Cart-Double-Pendulum-Direct-v0")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaaclab-multi-agent")'
# [jax-end-isaaclab-multi-agent]

# =============================================================================

# [pytorch-start-gym]
# import the environment wrapper and gym
from skrl.envs.wrappers.torch import wrap_env
import gym

# load the environment
env = gym.make('Pendulum-v1')

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'
# [pytorch-end-gym]


# [jax-start-gym]
# import the environment wrapper and gym
from skrl.envs.wrappers.jax import wrap_env
import gym

# load the environment
env = gym.make('Pendulum-v1')

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'
# [jax-end-gym]


# [pytorch-start-gym-vectorized]
# import the environment wrapper and gym
from skrl.envs.wrappers.torch import wrap_env
import gym

# load a vectorized environment
env = gym.vector.make("Pendulum-v1", num_envs=10, asynchronous=False)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'
# [pytorch-end-gym-vectorized]


# [jax-start-gym-vectorized]
# import the environment wrapper and gym
from skrl.envs.wrappers.jax import wrap_env
import gym

# load a vectorized environment
env = gym.vector.make("Pendulum-v1", num_envs=10, asynchronous=False)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'
# [jax-end-gym-vectorized]

# =============================================================================

# [pytorch-start-gymnasium]
# import the environment wrapper and gymnasium
from skrl.envs.wrappers.torch import wrap_env
import gymnasium as gym

# load the environment
env = gym.make('Pendulum-v1')

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [pytorch-end-gymnasium]


# [jax-start-gymnasium]
# import the environment wrapper and gymnasium
from skrl.envs.wrappers.jax import wrap_env
import gymnasium as gym

# load the environment
env = gym.make('Pendulum-v1')

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [jax-end-gymnasium]


# [pytorch-start-gymnasium-vectorized]
# import the environment wrapper and gymnasium
from skrl.envs.wrappers.torch import wrap_env
import gymnasium as gym

# load a vectorized environment
env = gym.make_vec("Pendulum-v1", num_envs=10, vectorization_mode="async")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [pytorch-end-gymnasium-vectorized]


# [jax-start-gymnasium-vectorized]
# import the environment wrapper and gymnasium
from skrl.envs.wrappers.jax import wrap_env
import gymnasium as gym

# load a vectorized environment
env = gym.make_vec("Pendulum-v1", num_envs=10, vectorization_mode="async")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [jax-end-gymnasium-vectorized]

# =============================================================================

# [pytorch-start-shimmy]
# import the environment wrapper and gymnasium
from skrl.envs.wrappers.torch import wrap_env
import gymnasium as gym

# load the environment (API conversion)
env = gym.make("ALE/Pong-v5")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [pytorch-end-shimmy]


# [jax-start-shimmy]
# import the environment wrapper and gymnasium
from skrl.envs.wrappers.jax import wrap_env
import gymnasium as gym

# load the environment (API conversion)
env = gym.make("ALE/Pong-v5")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [jax-end-shimmy]


# [pytorch-start-shimmy-multi-agent]
# import the environment wrapper
from skrl.envs.wrappers.torch import wrap_env

# import the shimmy module
from shimmy import MeltingPotCompatibilityV0

# load the environment (API conversion)
env = MeltingPotCompatibilityV0(substrate_name="prisoners_dilemma_in_the_matrix__arena")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="pettingzoo")'
# [pytorch-end-shimmy-multi-agent]


# [jax-start-shimmy-multi-agent]
# import the environment wrapper
from skrl.envs.wrappers.jax import wrap_env

# import the shimmy module
from shimmy import MeltingPotCompatibilityV0

# load the environment (API conversion)
env = MeltingPotCompatibilityV0(substrate_name="prisoners_dilemma_in_the_matrix__arena")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="pettingzoo")'
# [jax-end-shimmy-multi-agent]

# =============================================================================

# [pytorch-start-brax]
# import the environment wrapper
from skrl.envs.wrappers.torch import wrap_env
import brax.envs

# load the environment
env = brax.envs.create("inverted_pendulum", batch_size=4092, backend="spring")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="brax")'
# [pytorch-end-brax]


# [jax-start-brax]
# import the environment wrapper
from skrl.envs.wrappers.jax import wrap_env
import brax.envs

# load the environment
env = brax.envs.create("inverted_pendulum", batch_size=4092, backend="spring")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="brax")'
# [jax-end-brax]

# =============================================================================

# [pytorch-start-deepmind]
# import the environment wrapper and the deepmind suite
from skrl.envs.wrappers.torch import wrap_env
from dm_control import suite

# load the environment
env = suite.load(domain_name="cartpole", task_name="swingup")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="dm")'
# [pytorch-end-deepmind]

# =============================================================================

# [start-pettingzoo-torch]
# import the environment wrapper
from skrl.envs.wrappers.torch import wrap_env

# import a PettingZoo environment
from pettingzoo.sisl import multiwalker_v9

# load the environment
env = multiwalker_v9.parallel_env()

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="pettingzoo")'
# [end-pettingzoo-torch]


# [start-pettingzoo-jax]
# import the environment wrapper
from skrl.envs.wrappers.jax import wrap_env

# import a PettingZoo environment
from pettingzoo.sisl import multiwalker_v9

# load the environment
env = multiwalker_v9.parallel_env()

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="pettingzoo")'
# [end-pettingzoo-jax]
