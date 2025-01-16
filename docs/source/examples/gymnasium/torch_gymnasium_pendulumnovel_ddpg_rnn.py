import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# import the skrl components to build the RL system
from skrl.agents.torch.ddpg import DDPG_DEFAULT_CONFIG
from skrl.agents.torch.ddpg import DDPG_RNN as DDPG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (deterministic models) using mixin
class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=1, hidden_size=400, sequence_length=20):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size=self.num_observations,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.linear_layer_1 = nn.Linear(self.hidden_size, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            # get the hidden states corresponding to the initial sequence
            sequence_index = 1 if role == "target_policy" else 0  # target networks act on the next state of the environment
            hidden_states = hidden_states[:,:,sequence_index,:].contiguous()  # (D * num_layers, N, Hout)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, hidden_states = self.rnn(rnn_input[:,i0:i1,:], hidden_states)
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        x = F.relu(self.linear_layer_1(rnn_output))
        x = F.relu(self.linear_layer_2(x))

        # Pendulum-v1 action_space is -2 to 2
        return 2 * torch.tanh(self.action_layer(x)), {"rnn": [hidden_states]}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=1, hidden_size=400, sequence_length=20):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hout
        self.sequence_length = sequence_length

        self.rnn = nn.RNN(input_size=self.num_observations,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)

        self.linear_layer_1 = nn.Linear(self.hidden_size + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]

        # critic is only used during training
        rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

        hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
        # get the hidden states corresponding to the initial sequence
        sequence_index = 1 if role == "target_critic" else 0  # target networks act on the next state of the environment
        hidden_states = hidden_states[:,:,sequence_index,:].contiguous()  # (D * num_layers, N, Hout)

        # reset the RNN state in the middle of a sequence
        if terminated is not None and torch.any(terminated):
            rnn_outputs = []
            terminated = terminated.view(-1, self.sequence_length)
            indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

            for i in range(len(indexes) - 1):
                i0, i1 = indexes[i], indexes[i + 1]
                rnn_output, hidden_states = self.rnn(rnn_input[:,i0:i1,:], hidden_states)
                hidden_states[:, (terminated[:,i1-1]), :] = 0
                rnn_outputs.append(rnn_output)

            rnn_output = torch.cat(rnn_outputs, dim=1)
        # no need to reset the RNN state in the sequence
        else:
            rnn_output, hidden_states = self.rnn(rnn_input, hidden_states)

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        x = F.relu(self.linear_layer_1(torch.cat([rnn_output, inputs["taken_actions"]], dim=1)))
        x = F.relu(self.linear_layer_2(x))

        return self.linear_layer_3(x), {"rnn": [hidden_states]}


# environment observation wrapper used to mask velocity. Adapted from rl_zoo3 (rl_zoo3/wrappers.py)
class NoVelocityWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        # observation: x, y, angular velocity
        return observation * np.array([1, 1, 0])

gym.register(id="PendulumNoVel-v1", entry_point=lambda: NoVelocityWrapper(gym.make("Pendulum-v1")))

# load and wrap the gymnasium environment
env = gym.make("PendulumNoVel-v1")
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's models (function approximators).
# DDPG requires 4 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device, num_envs=env.num_envs)
models["target_policy"] = Actor(env.observation_space, env.action_space, device, num_envs=env.num_envs)
models["critic"] = Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs)
models["target_critic"] = Critic(env.observation_space, env.action_space, device, num_envs=env.num_envs)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
cfg = DDPG_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.1, base_scale=1.0, device=device)
cfg["discount_factor"] = 0.98
cfg["batch_size"] = 100
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 1000
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 75
cfg["experiment"]["checkpoint_interval"] = 750
cfg["experiment"]["directory"] = "runs/torch/PendulumNoVel"

agent = DDPG(models=models,
             memory=memory,
             cfg=cfg,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 15000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()
