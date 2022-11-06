from dm_control import manipulation

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env


# Define the models (stochastic and deterministic models) for the SAC agent using the mixins.
# - StochasticActor (policy): takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class StochasticActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.features_extractor = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=3),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=2, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                nn.Linear(7744, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 8),
                                                nn.Tanh())

        self.net = nn.Sequential(nn.Linear(26, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = inputs["states"]

        # The dm_control.manipulation tasks have as observation/state spec a `collections.OrderedDict` object as follows:
        # OrderedDict([('front_close', BoundedArray(shape=(1, 84, 84, 3), dtype=dtype('uint8'), name='front_close', minimum=0, maximum=255)),
        #              ('jaco_arm/joints_pos', Array(shape=(1, 6, 2), dtype=dtype('float64'), name='jaco_arm/joints_pos')),
        #              ('jaco_arm/joints_torque', Array(shape=(1, 6), dtype=dtype('float64'), name='jaco_arm/joints_torque')),
        #              ('jaco_arm/joints_vel', Array(shape=(1, 6), dtype=dtype('float64'), name='jaco_arm/joints_vel')),
        #              ('jaco_arm/jaco_hand/joints_pos', Array(shape=(1, 3), dtype=dtype('float64'), name='jaco_arm/jaco_hand/joints_pos')),
        #              ('jaco_arm/jaco_hand/joints_vel', Array(shape=(1, 3), dtype=dtype('float64'), name='jaco_arm/jaco_hand/joints_vel')),
        #              ('jaco_arm/jaco_hand/pinch_site_pos', Array(shape=(1, 3), dtype=dtype('float64'), name='jaco_arm/jaco_hand/pinch_site_pos')),
        #              ('jaco_arm/jaco_hand/pinch_site_rmat', Array(shape=(1, 9), dtype=dtype('float64'), name='jaco_arm/jaco_hand/pinch_site_rmat'))])

        # This spec is converted to a `gym.spaces.Dict` space by the `wrap_env` function as follows:
        # Dict(front_close: Box(0, 255, (1, 84, 84, 3), uint8),
        #      jaco_arm/jaco_hand/joints_pos: Box(-inf, inf, (1, 3), float64),
        #      jaco_arm/jaco_hand/joints_vel: Box(-inf, inf, (1, 3), float64),
        #      jaco_arm/jaco_hand/pinch_site_pos: Box(-inf, inf, (1, 3), float64),
        #      jaco_arm/jaco_hand/pinch_site_rmat: Box(-inf, inf, (1, 9), float64),
        #      jaco_arm/joints_pos: Box(-inf, inf, (1, 6, 2), float64),
        #      jaco_arm/joints_torque: Box(-inf, inf, (1, 6), float64),
        #      jaco_arm/joints_vel: Box(-inf, inf, (1, 6), float64))

        # The `spaces` parameter is a flat tensor of the flattened observation/state space with shape (batch_size, size_of_flat_space).
        # Using the model's method `tensor_to_space` we can convert the flattened tensor to the original space.
        # https://skrl.readthedocs.io/en/latest/modules/skrl.models.base_class.html#skrl.models.torch.base.Model.tensor_to_space
        space = self.tensor_to_space(states, self.observation_space)

        # For this case, the `space` variable is a Python dictionary with the following structure and shapes:
        # {'front_close': torch.Tensor(shape=[batch_size, 1, 84, 84, 3], dtype=torch.float32),
        #  'jaco_arm/jaco_hand/joints_pos': torch.Tensor(shape=[batch_size, 1, 3], dtype=torch.float32)
        #  'jaco_arm/jaco_hand/joints_vel': torch.Tensor(shape=[batch_size, 1, 3], dtype=torch.float32)
        #  'jaco_arm/jaco_hand/pinch_site_pos': torch.Tensor(shape=[batch_size, 1, 3], dtype=torch.float32)
        #  'jaco_arm/jaco_hand/pinch_site_rmat': torch.Tensor(shape=[batch_size, 1, 9], dtype=torch.float32)
        #  'jaco_arm/joints_pos': torch.Tensor(shape=[batch_size, 1, 6, 2], dtype=torch.float32)
        #  'jaco_arm/joints_torque': torch.Tensor(shape=[batch_size, 1, 6], dtype=torch.float32)
        #  'jaco_arm/joints_vel': torch.Tensor(shape=[batch_size, 1, 6], dtype=torch.float32)}

        # permute and normalize the images (samples, width, height, channels) -> (samples, channels, width, height)
        features = self.features_extractor(space['front_close'][:,0].permute(0, 3, 1, 2) / 255.0)

        mean_actions = torch.tanh(self.net(torch.cat([features,
                                                      space["jaco_arm/joints_pos"].view(states.shape[0], -1),
                                                      space["jaco_arm/joints_vel"].view(states.shape[0], -1)], dim=-1)))

        return mean_actions, self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.features_extractor = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=3),
                                                nn.ReLU(),
                                                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                                nn.ReLU(),
                                                nn.Conv2d(64, 64, kernel_size=2, stride=1),
                                                nn.ReLU(),
                                                nn.Flatten(),
                                                nn.Linear(7744, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 8),
                                                nn.Tanh())

        self.net = nn.Sequential(nn.Linear(26 + self.num_actions, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        states = inputs["states"]

        # map the observations/states to the original space.
        # See the explanation above (StochasticActor.compute)
        space = self.tensor_to_space(states, self.observation_space)

        # permute and normalize the images (samples, width, height, channels) -> (samples, channels, width, height)
        features = self.features_extractor(space['front_close'][:,0].permute(0, 3, 1, 2) / 255.0)

        return self.net(torch.cat([features,
                                   space["jaco_arm/joints_pos"].view(states.shape[0], -1),
                                   space["jaco_arm/joints_vel"].view(states.shape[0], -1),
                                   inputs["taken_actions"]], dim=-1)), {}


# Load and wrap the DeepMind environment
env = manipulation.load("reach_site_vision")
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory (without replacement) as experience replay memory
memory = RandomMemory(memory_size=50000, num_envs=env.num_envs, device=device, replacement=False)


# Instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#spaces-and-models
models_sac = {}
models_sac["policy"] = StochasticActor(env.observation_space, env.action_space, device, clip_actions=True)
models_sac["critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["critic_2"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models_sac["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

# Initialize the models' parameters (weights and biases) using a Gaussian distribution
for model in models_sac.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.sac.html#configuration-and-hyperparameters
cfg_sac = SAC_DEFAULT_CONFIG.copy()
cfg_sac["gradient_steps"] = 1
cfg_sac["batch_size"] = 256
cfg_sac["random_timesteps"] = 0
cfg_sac["learning_starts"] = 10000
cfg_sac["learn_entropy"] = True
# logging to TensorBoard and write checkpoints each 1000 and 5000 timesteps respectively
cfg_sac["experiment"]["write_interval"] = 1000
cfg_sac["experiment"]["checkpoint_interval"] = 5000


agent_sac = SAC(models=models_sac,
                memory=memory,
                cfg=cfg_sac,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_sac)

# start training
trainer.train()
