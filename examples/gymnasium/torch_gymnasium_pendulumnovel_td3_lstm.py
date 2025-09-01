import argparse
import os
import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.td3 import TD3_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3_RNN as TD3
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import GaussianNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no rendering)")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")
args, _ = parser.parse_known_args()


# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed


# define models (deterministic models) using mixin
class Actor(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        clip_actions=False,
        num_envs=1,
        num_layers=1,
        hidden_size=400,
        sequence_length=20,
    ):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(
            input_size=self.num_observations, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(
            nn.Linear(self.hidden_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, self.num_actions),
            nn.Tanh(),
        )

    def get_specification(self):
        # batch size (N) is the number of envs
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),  # hidden states (D ∗ num_layers, N, Hout)
                    (self.num_layers, self.num_envs, self.hidden_size),
                ],
            }
        }  # cell states (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        observations = inputs["observations"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            rnn_input = observations.view(
                -1, self.sequence_length, observations.shape[-1]
            )  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(
                self.num_layers, -1, self.sequence_length, hidden_states.shape[-1]
            )  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(
                self.num_layers, -1, self.sequence_length, cell_states.shape[-1]
            )  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            sequence_index = (
                1 if role == "target_policy" else 0
            )  # target networks act on the next state of the environment
            hidden_states = hidden_states[:, :, sequence_index, :].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:, :, sequence_index, :].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(
                        rnn_input[:, i0:i1, :], (hidden_states, cell_states)
                    )
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    cell_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = observations.view(-1, 1, observations.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        x = self.net(rnn_output)
        # Pendulum-v1 action_space is -2 to 2
        return 2.0 * x, {"rnn": [rnn_states[0], rnn_states[1]]}


class Critic(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        num_envs=1,
        num_layers=1,
        hidden_size=400,
        sequence_length=20,
    ):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(
            input_size=self.num_observations, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True
        )  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(
            nn.Linear(self.hidden_size + self.num_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def get_specification(self):
        # batch size (N) is the number of envs
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (self.num_layers, self.num_envs, self.hidden_size),  # hidden states (D ∗ num_layers, N, Hout)
                    (self.num_layers, self.num_envs, self.hidden_size),
                ],
            }
        }  # cell states (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        observations = inputs["observations"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # critic is only used during training
        rnn_input = observations.view(
            -1, self.sequence_length, observations.shape[-1]
        )  # (N, L, Hin): N=batch_size, L=sequence_length

        hidden_states = hidden_states.view(
            self.num_layers, -1, self.sequence_length, hidden_states.shape[-1]
        )  # (D * num_layers, N, L, Hout)
        cell_states = cell_states.view(
            self.num_layers, -1, self.sequence_length, cell_states.shape[-1]
        )  # (D * num_layers, N, L, Hcell)
        # get the hidden/cell states corresponding to the initial sequence
        sequence_index = (
            1 if role in ["target_critic_1", "target_critic_2"] else 0
        )  # target networks act on the next state of the environment
        hidden_states = hidden_states[:, :, sequence_index, :].contiguous()  # (D * num_layers, N, Hout)
        cell_states = cell_states[:, :, sequence_index, :].contiguous()  # (D * num_layers, N, Hcell)

        # reset the RNN state in the middle of a sequence
        if terminated is not None and torch.any(terminated):
            rnn_outputs = []
            terminated = terminated.view(-1, self.sequence_length)
            indexes = (
                [0] + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]
            )

            for i in range(len(indexes) - 1):
                i0, i1 = indexes[i], indexes[i + 1]
                rnn_output, (hidden_states, cell_states) = self.lstm(
                    rnn_input[:, i0:i1, :], (hidden_states, cell_states)
                )
                hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                cell_states[:, (terminated[:, i1 - 1]), :] = 0
                rnn_outputs.append(rnn_output)

            rnn_states = (hidden_states, cell_states)
            rnn_output = torch.cat(rnn_outputs, dim=1)
        # no need to reset the RNN state in the sequence
        else:
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        x = self.net(torch.cat([rnn_output, inputs["taken_actions"]], dim=1))
        return x, {"rnn": [rnn_states[0], rnn_states[1]]}


# environment observation wrapper used to mask velocity
class NoVelocityWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        # observation: x, y, angular velocity
        return observation * np.array([1, 1, 0], dtype=observation.dtype)


# register a custom environment
task_name = "PendulumNoVel"
env_id = [spec for spec in gym.envs.registry if spec.startswith("Pendulum-v")][-1]  # get latest environment version
gym.register(
    id=f"{task_name}-v1",
    entry_point=lambda *args, **kwargs: NoVelocityWrapper(gym.make(env_id, *args, **kwargs)),
)
# load the environment (note: the environment version may change depending on the gymnasium version)
render_mode = "human" if not args.headless else None
if args.num_envs <= 1:
    env = gym.make(f"{task_name}-v1", render_mode=render_mode)
else:
    env = gym.make_vec(f"{task_name}-v1", num_envs=args.num_envs, render_mode=render_mode, vectorization_mode="sync")
# wrap the environment
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=15000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's models (function approximators).
# TD3 requires 6 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/td3.html#models
models = {}
models["policy"] = Actor(env.observation_space, env.state_space, env.action_space, device, num_envs=env.num_envs)
models["target_policy"] = Actor(env.observation_space, env.state_space, env.action_space, device, num_envs=env.num_envs)
models["critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device, num_envs=env.num_envs)
models["critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device, num_envs=env.num_envs)
models["target_critic_1"] = Critic(
    env.observation_space, env.state_space, env.action_space, device, num_envs=env.num_envs
)
models["target_critic_2"] = Critic(
    env.observation_space, env.state_space, env.action_space, device, num_envs=env.num_envs
)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/td3.html#configuration-and-hyperparameters
cfg = TD3_DEFAULT_CONFIG.copy()
cfg["exploration"]["noise"] = GaussianNoise
cfg["exploration"]["noise_kwargs"] = {"mean": 0.0, "std": 0.1, "device": device}
cfg["smooth_regularization_noise"] = GaussianNoise
cfg["smooth_regularization_noise_kwargs"] = {"mean": 0.0, "std": 0.2, "device": device}
cfg["smooth_regularization_clip"] = 0.5
cfg["discount_factor"] = 0.98
cfg["batch_size"] = 100
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 100
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = "auto" if not args.eval else 0
cfg["experiment"]["checkpoint_interval"] = "auto" if not args.eval else 0
cfg["experiment"]["directory"] = f"runs/torch/{task_name}"

agent = TD3(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 15000, "headless": args.headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
