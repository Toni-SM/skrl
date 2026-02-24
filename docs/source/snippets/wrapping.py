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


# [warp-start-isaaclab]
# import the environment wrapper and loader
from skrl.envs.wrappers.warp import wrap_env
from skrl.envs.loaders.warp import load_isaaclab_env

# load the environment
env = load_isaaclab_env(task_name="Isaac-Cartpole-Direct-v0")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaaclab")'
# [warp-end-isaaclab]


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

# [pytorch-start-mani-skill]
# import the environment wrapper, gymnasium and mani_skill
from skrl.envs.wrappers.torch import wrap_env
import gymnasium as gym
import mani_skill.envs  # needed to register the ManiSkill environment entry points

# load the environment
env_kwargs = {"obs_mode": "state", "sim_backend": "physx_cuda", "control_mode": "pd_joint_delta_pos"}
env = gym.make("PushCube", num_envs=1024, **env_kwargs)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="mani-skill")'
# [pytorch-end-mani-skill]


# [jax-start-mani-skill]
# import the environment wrapper, gymnasium and mani_skill
from skrl.envs.wrappers.jax import wrap_env
import gymnasium as gym
import mani_skill.envs  # needed to register the ManiSkill environment entry points

# load the environment
env_kwargs = {"obs_mode": "state", "sim_backend": "physx_cuda", "control_mode": "pd_joint_delta_pos"}
env = gym.make("PushCube", num_envs=1024, **env_kwargs)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="mani-skill")'
# [jax-end-mani-skill]


# [warp-start-mani-skill]
# import the environment wrapper, gymnasium and mani_skill
from skrl.envs.wrappers.warp import wrap_env
import gymnasium as gym
import mani_skill.envs  # needed to register the ManiSkill environment entry points

# load the environment
env_kwargs = {"obs_mode": "state", "sim_backend": "physx_cuda", "control_mode": "pd_joint_delta_pos"}
env = gym.make("PushCube", num_envs=1024, **env_kwargs)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="mani-skill")'
# [warp-end-mani-skill]

# =============================================================================

# [pytorch-start-playground]
# import the environment wrapper and loader
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.loaders.torch import load_playground_env

# load the environment
env = load_playground_env(task_name="CartpoleBalance", num_envs=1024, episode_length=300)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="playground")'
# [pytorch-end-playground]


# [jax-start-playground]
# import the environment wrapper and loader
from skrl.envs.wrappers.jax import wrap_env
from skrl.envs.loaders.jax import load_playground_env

# load the environment
env = load_playground_env(task_name="CartpoleBalance", num_envs=1024, episode_length=300)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="playground")'
# [jax-end-playground]


# [warp-start-playground]
# import the environment wrapper and loader
from skrl.envs.wrappers.warp import wrap_env
from skrl.envs.loaders.warp import load_playground_env

# load the environment
env = load_playground_env(task_name="CartpoleBalance", num_envs=1024, episode_length=300)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="playground")'
# [warp-end-playground]

# =============================================================================

# [pytorch-start-gym]
# import the environment wrapper and gym
from skrl.envs.wrappers.torch import wrap_env
import gym

# load the environment
env = gym.make("Pendulum-v1")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'
# [pytorch-end-gym]


# [jax-start-gym]
# import the environment wrapper and gym
from skrl.envs.wrappers.jax import wrap_env
import gym

# load the environment
env = gym.make("Pendulum-v1")

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
env = gym.make("Pendulum-v1")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [pytorch-end-gymnasium]


# [jax-start-gymnasium]
# import the environment wrapper and gymnasium
from skrl.envs.wrappers.jax import wrap_env
import gymnasium as gym

# load the environment
env = gym.make("Pendulum-v1")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [jax-end-gymnasium]


# [warp-start-gymnasium]
# import the environment wrapper and gymnasium
from skrl.envs.wrappers.warp import wrap_env
import gymnasium as gym

# load the environment
env = gym.make("Pendulum-v1")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [warp-end-gymnasium]


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


# [warp-start-gymnasium-vectorized]
# import the environment wrapper and gymnasium
from skrl.envs.wrappers.warp import wrap_env
import gymnasium as gym

# load a vectorized environment
env = gym.make_vec("Pendulum-v1", num_envs=10, vectorization_mode="async")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [warp-end-gymnasium-vectorized]

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


# [warp-start-shimmy]
# import the environment wrapper and gymnasium
from skrl.envs.wrappers.warp import wrap_env
import gymnasium as gym

# load the environment (API conversion)
env = gym.make("ALE/Pong-v5")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [warp-end-shimmy]


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
