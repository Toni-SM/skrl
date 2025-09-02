import argparse
import os
import gym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise
from skrl.trainers.torch import SequentialTrainer, generate_equally_spaced_scopes
from skrl.utils import set_seed


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=3, help="Number of environments")
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no rendering)")
parser.add_argument("--seed", type=int, default=None, help="Random seed")
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")
args, _ = parser.parse_known_args()


# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed


# define models using mixins
class StochasticActor(GaussianMixin, Model):
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

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, self.num_actions),
            nn.Tanh(),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = self.net(inputs["observations"])
        # Pendulum-v1 action_space is -2 to 2
        return 2.0 * x, {"log_std": self.log_std_parameter}


class DeterministicActor(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device, clip_actions=False):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self, clip_actions=clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, self.num_actions),
            nn.Tanh(),
        )

    def compute(self, inputs, role):
        x = self.net(inputs["observations"])
        # Pendulum-v1 action_space is -2 to 2
        return 2.0 * x, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self, observation_space=observation_space, state_space=state_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations + self.num_actions, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def compute(self, inputs, role):
        x = self.net(torch.cat([inputs["observations"], inputs["taken_actions"]], dim=1))
        return x, {}


if __name__ == "__main__":

    # load the environment (note: the environment version may change depending on the gym version)
    task_name = "Pendulum"
    render_mode = "human" if not args.headless else None
    env_id = [spec for spec in gym.envs.registry if spec.startswith(f"{task_name}-v")][
        -1
    ]  # get latest environment version
    if args.num_envs <= 1:
        env = gym.make(env_id, render_mode=render_mode)
    else:
        env = gym.vector.make(env_id, num_envs=args.num_envs, render_mode=render_mode, asynchronous=False)
    # wrap the environment
    env = wrap_env(env)

    device = env.device
    scopes = generate_equally_spaced_scopes(num_envs=args.num_envs, num_simultaneous_agents=3)

    # instantiate a memory as experience replay (shared by all agents)
    memory = RandomMemory(memory_size=15000, num_envs=env.num_envs, device=device)

    # instantiate the agents' models (function approximators).
    # DDPG requires 4 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#models
    models_ddpg = {}
    models_ddpg["policy"] = DeterministicActor(env.observation_space, env.state_space, env.action_space, device)
    models_ddpg["target_policy"] = DeterministicActor(env.observation_space, env.state_space, env.action_space, device)
    models_ddpg["critic"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_ddpg["target_critic"] = Critic(env.observation_space, env.state_space, env.action_space, device)

    # TD3 requires 6 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/td3.html#models
    models_td3 = {}
    models_td3["policy"] = DeterministicActor(env.observation_space, env.state_space, env.action_space, device)
    models_td3["target_policy"] = DeterministicActor(env.observation_space, env.state_space, env.action_space, device)
    models_td3["critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_td3["critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_td3["target_critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_td3["target_critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)

    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
    models_sac = {}
    models_sac["policy"] = StochasticActor(env.observation_space, env.state_space, env.action_space, device)
    models_sac["critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_sac["critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_sac["target_critic_1"] = Critic(env.observation_space, env.state_space, env.action_space, device)
    models_sac["target_critic_2"] = Critic(env.observation_space, env.state_space, env.action_space, device)

    # configure and instantiate the agents (visit their documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ddpg.html#configuration-and-hyperparameters
    cfg_ddpg = DDPG_DEFAULT_CONFIG.copy()
    cfg_ddpg["exploration"]["noise"] = OrnsteinUhlenbeckNoise
    cfg_ddpg["exploration"]["noise_kwargs"] = {"theta": 0.15, "sigma": 0.1, "base_scale": 1.0, "device": device}
    cfg_ddpg["batch_size"] = 100
    cfg_ddpg["random_timesteps"] = 100
    cfg_ddpg["learning_starts"] = 100
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg_ddpg["experiment"]["write_interval"] = "auto" if not args.eval else 0
    cfg_ddpg["experiment"]["checkpoint_interval"] = "auto" if not args.eval else 0
    cfg_ddpg["experiment"]["directory"] = f"runs/torch/{task_name}-DDPG"

    # https://skrl.readthedocs.io/en/latest/api/agents/td3.html#configuration-and-hyperparameters
    cfg_td3 = TD3_DEFAULT_CONFIG.copy()
    cfg_td3["exploration"]["noise"] = GaussianNoise
    cfg_td3["exploration"]["noise_kwargs"] = {"mean": 0.0, "std": 0.1, "device": device}
    cfg_td3["smooth_regularization_noise"] = GaussianNoise
    cfg_td3["smooth_regularization_noise_kwargs"] = {"mean": 0.0, "std": 0.2, "device": device}
    cfg_td3["smooth_regularization_clip"] = 0.5
    cfg_td3["discount_factor"] = 0.98
    cfg_td3["batch_size"] = 100
    cfg_td3["random_timesteps"] = 100
    cfg_td3["learning_starts"] = 100
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg_td3["experiment"]["write_interval"] = "auto" if not args.eval else 0
    cfg_td3["experiment"]["checkpoint_interval"] = "auto" if not args.eval else 0
    cfg_td3["experiment"]["directory"] = f"runs/torch/{task_name}-TD3"

    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#configuration-and-hyperparameters
    cfg_sac = SAC_DEFAULT_CONFIG.copy()
    cfg_sac["discount_factor"] = 0.98
    cfg_sac["batch_size"] = 100
    cfg_sac["random_timesteps"] = 0
    cfg_sac["learning_starts"] = 100
    cfg_sac["learn_entropy"] = True
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg_sac["experiment"]["write_interval"] = "auto" if not args.eval else 0
    cfg_sac["experiment"]["checkpoint_interval"] = "auto" if not args.eval else 0
    cfg_sac["experiment"]["directory"] = f"runs/torch/{task_name}-SAC"

    agent_ddpg = DDPG(
        models=models_ddpg,
        memory=memory,
        cfg=cfg_ddpg,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
    )

    agent_td3 = TD3(
        models=models_td3,
        memory=memory,
        cfg=cfg_td3,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
    )

    agent_sac = SAC(
        models=models_sac,
        memory=memory,
        cfg=cfg_sac,
        observation_space=env.observation_space,
        state_space=env.state_space,
        action_space=env.action_space,
        device=device,
    )

    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 15000, "headless": args.headless}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent_ddpg, agent_td3, agent_sac], scopes=scopes)

    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
            exit(1)
        raise NotImplementedError("The logic for loading checkpoints for each agent is not implemented in this example")

    trainer.train() if not args.eval else trainer.eval()
