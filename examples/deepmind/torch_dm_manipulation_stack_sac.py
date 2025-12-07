import argparse
import os
from dm_control import manipulation

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no rendering)")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")
args, _ = parser.parse_known_args()


# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
class Actor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        state_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        GaussianMixin.__init__(
            self,
            clip_actions=clip_actions,
            clip_log_std=clip_log_std,
            min_log_std=min_log_std,
            max_log_std=max_log_std,
            reduction=reduction,
        )

        self.features_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7744, 512),
            nn.ReLU(),
            nn.Linear(512, 8),
            nn.Tanh(),
        )

        self.net = nn.Sequential(
            nn.Linear(26, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
        )

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # dm_control.manipulation observation spec is a `collections.OrderedDict` object:
        # OrderedDict([
        #     ('front_close', BoundedArray(shape=(1, 84, 84, 3), dtype=dtype('uint8'), name='front_close', minimum=0, maximum=255)),
        #     ('jaco_arm/joints_pos', Array(shape=(1, 6, 2), dtype=dtype('float64'), name='jaco_arm/joints_pos')),
        #     ('jaco_arm/joints_torque', Array(shape=(1, 6), dtype=dtype('float64'), name='jaco_arm/joints_torque')),
        #     ('jaco_arm/joints_vel', Array(shape=(1, 6), dtype=dtype('float64'), name='jaco_arm/joints_vel')),
        #     ('jaco_arm/jaco_hand/joints_pos', Array(shape=(1, 3), dtype=dtype('float64'), name='jaco_arm/jaco_hand/joints_pos')),
        #     ('jaco_arm/jaco_hand/joints_vel', Array(shape=(1, 3), dtype=dtype('float64'), name='jaco_arm/jaco_hand/joints_vel')),
        #     ('jaco_arm/jaco_hand/pinch_site_pos', Array(shape=(1, 3), dtype=dtype('float64'), name='jaco_arm/jaco_hand/pinch_site_pos')),
        #     ('jaco_arm/jaco_hand/pinch_site_rmat', Array(shape=(1, 9), dtype=dtype('float64'), name='jaco_arm/jaco_hand/pinch_site_rmat')),
        # ])

        # observation spec converted to a `gymnasium.spaces.Dict` object by the wrapper:
        # Dict(
        #     'front_close': Box(0, 255, (1, 84, 84, 3), uint8),
        #     'jaco_arm/jaco_hand/joints_pos': Box(-inf, inf, (1, 3), float64),
        #     'jaco_arm/jaco_hand/joints_vel': Box(-inf, inf, (1, 3), float64),
        #     'jaco_arm/jaco_hand/pinch_site_pos': Box(-inf, inf, (1, 3), float64),
        #     'jaco_arm/jaco_hand/pinch_site_rmat': Box(-inf, inf, (1, 9), float64),
        #     'jaco_arm/joints_pos': Box(-inf, inf, (1, 6, 2), float64),
        #     'jaco_arm/joints_torque': Box(-inf, inf, (1, 6), float64),
        #     'jaco_arm/joints_vel': Box(-inf, inf, (1, 6), float64),
        # )

        # using the space utility, the `inputs.get("observations")`, a flattened tensor
        # with shape (batch_size, space_size), is converted to the original space format
        # https://skrl.readthedocs.io/en/latest/api/utils/spaces.html
        batch_size = inputs.get("observations").shape[0]
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))

        # for this case, the `space` variable is a Python dictionary with the following structure and shapes:
        # {
        #     'front_close': torch.Tensor(shape=[batch_size, 1, 84, 84, 3], dtype=torch.float32),
        #     'jaco_arm/jaco_hand/joints_pos': torch.Tensor(shape=[batch_size, 1, 3], dtype=torch.float32)
        #     'jaco_arm/jaco_hand/joints_vel': torch.Tensor(shape=[batch_size, 1, 3], dtype=torch.float32)
        #     'jaco_arm/jaco_hand/pinch_site_pos': torch.Tensor(shape=[batch_size, 1, 3], dtype=torch.float32)
        #     'jaco_arm/jaco_hand/pinch_site_rmat': torch.Tensor(shape=[batch_size, 1, 9], dtype=torch.float32)
        #     'jaco_arm/joints_pos': torch.Tensor(shape=[batch_size, 1, 6, 2], dtype=torch.float32)
        #     'jaco_arm/joints_torque': torch.Tensor(shape=[batch_size, 1, 6], dtype=torch.float32)
        #     'jaco_arm/joints_vel': torch.Tensor(shape=[batch_size, 1, 6], dtype=torch.float32)
        # }

        # permute and normalize the images (samples, width, height, channels) -> (samples, channels, width, height)
        features = self.features_extractor(observations["front_close"][:, 0].permute(0, 3, 1, 2) / 255.0)

        x = self.net(
            torch.cat(
                [
                    features,
                    observations["jaco_arm/joints_pos"].view(batch_size, -1),
                    observations["jaco_arm/joints_vel"].view(batch_size, -1),
                ],
                dim=-1,
            )
        )

        return torch.tanh(x), {"log_std": self.log_std_parameter}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self)

        self.features_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7744, 512),
            nn.ReLU(),
            nn.Linear(512, 8),
            nn.Tanh(),
        )

        self.net = nn.Sequential(
            nn.Linear(26 + self.num_actions, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def compute(self, inputs, role):
        # map the observations to the original space (see the explanation above: Actor.compute)
        batch_size = inputs.get("observations").shape[0]
        observations = unflatten_tensorized_space(self.observation_space, inputs.get("observations"))

        # permute and normalize the images (samples, width, height, channels) -> (samples, channels, width, height)
        features = self.features_extractor(observations["front_close"][:, 0].permute(0, 3, 1, 2) / 255.0)

        x = self.net(
            torch.cat(
                [
                    features,
                    observations["jaco_arm/joints_pos"].view(batch_size, -1),
                    observations["jaco_arm/joints_vel"].view(batch_size, -1),
                    inputs["taken_actions"],
                ],
                dim=-1,
            )
        )
        return x, {}


# load the environment
task_name = "reach_site_vision"
env = manipulation.load(task_name)
# wrap the environment
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=50000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's models (function approximators).
# SAC requires 5 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
models = {}
models["policy"] = Actor(env.observation_space, env.state_space, env.action_space, device)
models["critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
cfg = SAC_DEFAULT_CONFIG.copy()
cfg["gradient_steps"] = 1
cfg["batch_size"] = 256
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 1000
cfg["learn_entropy"] = True
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = "auto" if not args.eval else 0
cfg["experiment"]["checkpoint_interval"] = "auto" if not args.eval else 0
cfg["experiment"]["directory"] = f"runs/torch/{task_name}"

agent = SAC(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 100000, "headless": args.headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
