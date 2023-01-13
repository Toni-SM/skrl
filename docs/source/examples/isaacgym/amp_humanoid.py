import isaacgym

import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
from skrl.envs.torch import load_isaacgym_env_preview4
from skrl.utils import set_seed


# set the seed for reproducibility
set_seed(42)


# Define the models (stochastic and deterministic models) for the agent using mixins.
# - Policy: takes as input the environment's observation/state and returns an action
# - Value: takes the state as input and provides a value to guide the policy
# - Discriminator: differentiate between police-generated behaviors and behaviors from the motion dataset
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, self.num_actions))

        # set a fixed log standard deviation for the policy
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), fill_value=-2.9), requires_grad=False)

    def compute(self, inputs, role):
        return torch.tanh(self.net(inputs["states"])), self.log_std_parameter, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

class Discriminator(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 1024),
                                 nn.ReLU(),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# Load and wrap the Isaac Gym environment
env = load_isaacgym_env_preview4(task_name="HumanoidAMP")   # preview 3 and 4 use the same loader
env = wrap_env(env)

device = env.device


# Instantiate a RandomMemory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# Instantiate the agent's models (function approximators).
# AMP requires 3 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.amp.html#spaces-and-models
models_amp = {}
models_amp["policy"] = Policy(env.observation_space, env.action_space, device)
models_amp["value"] = Value(env.observation_space, env.action_space, device)
models_amp["discriminator"] = Discriminator(env.amp_observation_space, env.action_space, device)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.amp.html#configuration-and-hyperparameters
cfg_amp = AMP_DEFAULT_CONFIG.copy()
cfg_amp["rollouts"] = 16  # memory_size
cfg_amp["learning_epochs"] = 6
cfg_amp["mini_batches"] = 2  # 16 * 4096 / 32768
cfg_amp["discount_factor"] = 0.99
cfg_amp["lambda"] = 0.95
cfg_amp["learning_rate"] = 5e-5
cfg_amp["random_timesteps"] = 0
cfg_amp["learning_starts"] = 0
cfg_amp["grad_norm_clip"] = 0.0
cfg_amp["ratio_clip"] = 0.2
cfg_amp["value_clip"] = 0.2
cfg_amp["clip_predicted_values"] = False
cfg_amp["entropy_loss_scale"] = 0.0
cfg_amp["value_loss_scale"] = 2.5
cfg_amp["discriminator_loss_scale"] = 5.0
cfg_amp["amp_batch_size"] = 512
cfg_amp["task_reward_weight"] = 0.0
cfg_amp["style_reward_weight"] = 1.0
cfg_amp["discriminator_batch_size"] = 4096
cfg_amp["discriminator_reward_scale"] = 2
cfg_amp["discriminator_logit_regularization_scale"] = 0.05
cfg_amp["discriminator_gradient_penalty_scale"] = 5
cfg_amp["discriminator_weight_decay_scale"] = 0.0001
cfg_amp["state_preprocessor"] = RunningStandardScaler
cfg_amp["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg_amp["value_preprocessor"] = RunningStandardScaler
cfg_amp["value_preprocessor_kwargs"] = {"size": 1, "device": device}
cfg_amp["amp_state_preprocessor"] = RunningStandardScaler
cfg_amp["amp_state_preprocessor_kwargs"] = {"size": env.amp_observation_space, "device": device}
# logging to TensorBoard and write checkpoints each 16 and 4000 timesteps respectively
cfg_amp["experiment"]["write_interval"] = 160
cfg_amp["experiment"]["checkpoint_interval"] = 4000

agent = AMP(models=models_amp,
            memory=memory,
            cfg=cfg_amp,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device,
            amp_observation_space=env.amp_observation_space,
            motion_dataset=RandomMemory(memory_size=200000, device=device),
            reply_buffer=RandomMemory(memory_size=1000000, device=device),
            collect_reference_motions=lambda num_samples: env.fetch_amp_obs_demo(num_samples),
            collect_observation=lambda: env.reset_done()[0]["obs"])


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 80000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
