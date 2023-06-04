# [pytorch-start-omniverse-isaacgym]
# import the environment wrapper and loader
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_omniverse_isaacgym_env

# load the environment
env = load_omniverse_isaacgym_env(task_name="Cartpole")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="omniverse-isaacgym")'
# [pytorch-end-omniverse-isaacgym]


# [pytorch-start-omniverse-isaacgym-mt]
# import the environment wrapper and loader
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_omniverse_isaacgym_env

# load the multi-threaded environment
env = load_omniverse_isaacgym_env(task_name="Cartpole", multi_threaded=True, timeout=30)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="omniverse-isaacgym")'
# [pytorch-end-omniverse-isaacgym-mt]


# [pytorch-start-isaac-orbit]
# import the environment wrapper and loader
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaac_orbit_env

# load the environment
env = load_isaac_orbit_env(task_name="Isaac-Cartpole-v0")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaac-orbit")'
# [pytorch-end-isaac-orbit]


# [pytorch-start-isaacgym-preview4-make]
import isaacgymenvs

# import the environment wrapper
from skrl.envs.torch import wrap_env

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


# [pytorch-start-isaacgym-preview4]
# import the environment wrapper and loader
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview4

# load the environment
env = load_isaacgym_env_preview4(task_name="Cartpole")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview4")'
# [pytorch-end-isaacgym-preview4]


# [pytorch-start-isaacgym-preview3]
# import the environment wrapper and loader
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview3

# load the environment
env = load_isaacgym_env_preview3(task_name="Cartpole")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview3")'
# [pytorch-end-isaacgym-preview3]


# [pytorch-start-isaacgym-preview2]
# import the environment wrapper and loader
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview2

# load the environment
env = load_isaacgym_env_preview2(task_name="Cartpole")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview2")'
# [pytorch-end-isaacgym-preview2]


# [pytorch-start-gym]
# import the environment wrapper and gym
from skrl.envs.torch import wrap_env
import gym

# load environment
env = gym.make('Pendulum-v1')

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'
# [pytorch-end-gym]


# [pytorch-start-gym-vectorized]
# import the environment wrapper and gym
from skrl.envs.torch import wrap_env
import gym

# load a vectorized environment
env = gym.vector.make("Pendulum-v1", num_envs=10, asynchronous=False)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'
# [pytorch-end-gym-vectorized]


# [pytorch-start-gymnasium]
# import the environment wrapper and gymnasium
from skrl.envs.torch import wrap_env
import gymnasium as gym

# load environment
env = gym.make('Pendulum-v1')

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [pytorch-end-gymnasium]


# [pytorch-start-gymnasium-vectorized]
# import the environment wrapper and gymnasium
from skrl.envs.torch import wrap_env
import gymnasium as gym

# load a vectorized environment
env = gym.vector.make("Pendulum-v1", num_envs=10, asynchronous=False)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'
# [pytorch-end-gymnasium-vectorized]


# [pytorch-start-deepmind]
# import the environment wrapper and the deepmind suite
from skrl.envs.torch import wrap_env
from dm_control import suite

# load environment
env = suite.load(domain_name="cartpole", task_name="swingup")

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="dm")'
# [pytorch-end-deepmind]


# [pytorch-start-robosuite]
# import the environment wrapper and robosuite
from skrl.envs.torch import wrap_env
import robosuite
from robosuite.controllers import load_controller_config

# load environment
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