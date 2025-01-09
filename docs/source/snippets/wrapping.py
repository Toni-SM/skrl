# [pytorch-start-omniverse-isaacgym]
# import the environment wrapper and loader
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env

# load the environment
env = load_omniverse_isaacgym_env(task_name="Cartpole")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="omniverse-isaacgym")'
# [pytorch-end-omniverse-isaacgym]

# [jax-start-omniverse-isaacgym]
# import the environment wrapper and loader
from skrl.envs.wrappers.jax import wrap_env
from skrl.envs.loaders.jax import load_omniverse_isaacgym_env

# load the environment
env = load_omniverse_isaacgym_env(task_name="Cartpole")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="omniverse-isaacgym")'
# [jax-end-omniverse-isaacgym]


# [pytorch-start-omniverse-isaacgym-mt]
# import the environment wrapper and loader
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env

# load the multi-threaded environment
env = load_omniverse_isaacgym_env(task_name="Cartpole", multi_threaded=True, timeout=30)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="omniverse-isaacgym")'
# [pytorch-end-omniverse-isaacgym-mt]


# [jax-start-omniverse-isaacgym-mt]
# import the environment wrapper and loader
from skrl.envs.wrappers.jax import wrap_env
from skrl.envs.loaders.jax import load_omniverse_isaacgym_env

# load the multi-threaded environment
env = load_omniverse_isaacgym_env(task_name="Cartpole", multi_threaded=True, timeout=30)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="omniverse-isaacgym")'
# [jax-end-omniverse-isaacgym-mt]

# =============================================================================

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

# [pytorch-start-isaacgym-preview4-make]
import isaacgymenvs

# import the environment wrapper
from skrl.envs.wrappers.torch import wrap_env

# create/load the environment using the easy-to-use API from NVIDIA
env = isaacgymenvs.make(seed=0,
                        task="Cartpole",
                        num_envs=512,
                        sim_device="cuda:0",
                        rl_device="cuda:0",
                        graphics_device_id=0,
                        headless=False)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview4")'
# [pytorch-end-isaacgym-preview4-make]


# [jax-start-isaacgym-preview4-make]
import isaacgymenvs

# import the environment wrapper
from skrl.envs.wrappers.jax import wrap_env

# create/load the environment using the easy-to-use API from NVIDIA
env = isaacgymenvs.make(seed=0,
                        task="Cartpole",
                        num_envs=512,
                        sim_device="cuda:0",
                        rl_device="cuda:0",
                        graphics_device_id=0,
                        headless=False)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview4")'
# [jax-end-isaacgym-preview4-make]


# [pytorch-start-isaacgym-preview4]
# import the environment wrapper and loader
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.loaders.torch import load_isaacgym_env_preview4

# load the environment
env = load_isaacgym_env_preview4(task_name="Cartpole")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview4")'
# [pytorch-end-isaacgym-preview4]


# [jax-start-isaacgym-preview4]
# import the environment wrapper and loader
from skrl.envs.wrappers.jax import wrap_env
from skrl.envs.loaders.jax import load_isaacgym_env_preview4

# load the environment
env = load_isaacgym_env_preview4(task_name="Cartpole")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview4")'
# [jax-end-isaacgym-preview4]

# =============================================================================

# [pytorch-start-isaacgym-preview3]
# import the environment wrapper and loader
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.loaders.torch import load_isaacgym_env_preview3

# load the environment
env = load_isaacgym_env_preview3(task_name="Cartpole")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview3")'
# [pytorch-end-isaacgym-preview3]


# [jax-start-isaacgym-preview3]
# import the environment wrapper and loader
from skrl.envs.wrappers.jax import wrap_env
from skrl.envs.loaders.jax import load_isaacgym_env_preview3

# load the environment
env = load_isaacgym_env_preview3(task_name="Cartpole")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview3")'
# [jax-end-isaacgym-preview3]

# =============================================================================

# [pytorch-start-isaacgym-preview2]
# import the environment wrapper and loader
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.loaders.torch import load_isaacgym_env_preview2

# load the environment
env = load_isaacgym_env_preview2(task_name="Cartpole")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview2")'
# [pytorch-end-isaacgym-preview2]


# [jax-start-isaacgym-preview2]
# import the environment wrapper and loader
from skrl.envs.wrappers.jax import wrap_env
from skrl.envs.loaders.jax import load_isaacgym_env_preview2

# load the environment
env = load_isaacgym_env_preview2(task_name="Cartpole")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview2")'
# [jax-end-isaacgym-preview2]

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
env = gym.vector.make("Pendulum-v1", num_envs=10, asynchronous=False)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [pytorch-end-gymnasium-vectorized]


# [jax-start-gymnasium-vectorized]
# import the environment wrapper and gymnasium
from skrl.envs.wrappers.jax import wrap_env
import gymnasium as gym

# load a vectorized environment
env = gym.vector.make("Pendulum-v1", num_envs=10, asynchronous=False)

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

# [pytorch-start-robosuite]
# import the environment wrapper
from skrl.envs.wrappers.torch import wrap_env

# import the robosuite modules
import robosuite
from robosuite.controllers import load_controller_config

# load the environment
controller_config = load_controller_config(default_controller="OSC_POSE")
env = robosuite.make("TwoArmLift",
                     robots=["Sawyer", "Panda"],             # load a Sawyer robot and a Panda robot
                     gripper_types="default",                # use default grippers per robot arm
                     controller_configs=controller_config,   # each arm is controlled using OSC
                     env_configuration="single-arm-opposed", # (two-arm envs only) arms face each other
                     has_renderer=True,                      # on-screen rendering
                     render_camera="frontview",              # visualize the "frontview" camera
                     has_offscreen_renderer=False,           # no off-screen rendering
                     control_freq=20,                        # 20 hz control for applied actions
                     horizon=200,                            # each episode terminates after 200 steps
                     use_object_obs=True,                    # provide object observations to agent
                     use_camera_obs=False,                   # don't provide image observations to agent
                     reward_shaping=True)                    # use a dense reward signal for learning

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="robosuite")'
# [pytorch-end-robosuite]

# =============================================================================

# [start-bidexhands-torch]
# import the environment wrapper and loader
from skrl.envs.wrappers.torch import wrap_env
from skrl.envs.loaders.torch import load_bidexhands_env

# load the environment
env = load_bidexhands_env(task_name="ShadowHandOver")

# wrap the environment
env = wrap_env(env, wrapper="bidexhands")
# [end-bidexhands-torch]


# [start-bidexhands-jax]
# import the environment wrapper and loader
from skrl.envs.wrappers.jax import wrap_env
from skrl.envs.loaders.jax import load_bidexhands_env

# load the environment
env = load_bidexhands_env(task_name="ShadowHandOver")

# wrap the environment
env = wrap_env(env, wrapper="bidexhands")
# [end-bidexhands-jax]

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
