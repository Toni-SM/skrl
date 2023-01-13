import robosuite
from robosuite.controllers import load_controller_config

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the skrl components to build the RL system
from skrl.models.torch import Model, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.resources.noises.torch import GaussianNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env


# Define the models (deterministic models) for the TD3 agent using mixins
# and programming with two approaches (torch functional and torch.nn.Sequential class).
# - Actor (policy): takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return torch.tanh(self.action_layer(x)), {}

class DeterministicCritic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations + self.num_actions, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 300),
                                 nn.ReLU(),
                                 nn.Linear(300, 1))

    def compute(self, inputs, role):
        return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}


# Load and wrap the DeepMind robosuite environment
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
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=25000, num_envs=env.num_envs, device=device, replacement=False)


# Instantiate the agent's models (function approximators).
# TD3 requires 6 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#spaces-and-models
models = {}
models["policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["target_policy"] = DeterministicActor(env.observation_space, env.action_space, device)
models["critic_1"] = DeterministicCritic(env.observation_space, env.action_space, device)
models["critic_2"] = DeterministicCritic(env.observation_space, env.action_space, device)
models["target_critic_1"] = DeterministicCritic(env.observation_space, env.action_space, device)
models["target_critic_2"] = DeterministicCritic(env.observation_space, env.action_space, device)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.td3.html#configuration-and-hyperparameters
cfg_agent = TD3_DEFAULT_CONFIG.copy()
cfg_agent["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
cfg_agent["smooth_regularization_noise"] = GaussianNoise(0, 0.2, device=device)
cfg_agent["smooth_regularization_clip"] = 0.5
cfg_agent["batch_size"] = 100
cfg_agent["random_timesteps"] = 100
cfg_agent["learning_starts"] = 100
# logging to TensorBoard and write checkpoints each 1000 and 5000 timesteps respectively
cfg_agent["experiment"]["write_interval"] = 1000
cfg_agent["experiment"]["checkpoint_interval"] = 5000

agent = TD3(models=models,
            memory=memory,
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50000, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
