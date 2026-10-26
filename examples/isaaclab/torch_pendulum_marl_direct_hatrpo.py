import argparse
import os

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.hatrpo import HATRPO, HATRPO_CFG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint from path")
parser.add_argument("--eval", action="store_true", help="Run in evaluation mode (logging/checkpointing disabled)")


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
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
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
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
            nn.Linear(self.num_observations, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["observations"]), {"log_std": self.log_std_parameter}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, state_space, action_space, device):
        Model.__init__(
            self,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
        )
        DeterministicMixin.__init__(self)

        self.net = nn.Sequential(
            nn.Linear(self.num_states, 32),
            nn.ELU(),
            nn.Linear(32, 32),
            nn.ELU(),
            nn.Linear(32, 1),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


# load the environment
task_name = "Isaac-Pendulum-MARL-Direct"
env = load_isaaclab_env(task_name=task_name, parser=parser)
# wrap the environment
env = wrap_env(env)

device = env.device


# defer parsing of arguments to include loader arguments (run with --help to see all the arguments)
args, _ = parser.parse_known_args()


# seed for reproducibility
set_seed(args.seed)  # e.g. `set_seed(42)` for fixed seed


# instantiate memories as rollout buffer (any memory can be used for this)
memories = {}
for agent_name in env.possible_agents:
    memories[agent_name] = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# HATRPO requires 2 models per agent, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/multi_agents/hatrpo.html#models
# - the value model (which takes the environment state) is shared by all agents: the same instance is used for each
value = Value(
    env.observation_space(env.possible_agents[0]),
    env.state_space(env.possible_agents[0]),
    env.action_space(env.possible_agents[0]),
    device,
)
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = Policy(
        env.observation_space(agent_name), env.state_space(agent_name), env.action_space(agent_name), device
    )
    models[agent_name]["value"] = value


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/multi_agents/hatrpo.html#configuration-and-hyperparameters
cfg = HATRPO_CFG()
cfg.rollouts = 16  # memory_size
# - the policies are updated with a trust region step on the whole rollout: the learning epochs, mini-batches,
#   learning rate and gradient clipping only apply to the value network
cfg.learning_epochs = 5
cfg.mini_batches = 1
cfg.discount_factor = 0.99
cfg.gae_lambda = 0.95
# - a high value learning rate (the policy steps are bounded by the KL divergence constraint): with 1e-3, the value
#   network learns too slowly and the training often gets stuck in a local optimum
cfg.learning_rate = 1e-2
cfg.observation_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {
    agent_name: {"size": env.observation_space(agent_name), "device": device} for agent_name in env.possible_agents
}
cfg.state_preprocessor = RunningStandardScaler
cfg.state_preprocessor_kwargs = {"size": env.state_space(env.possible_agents[0]), "device": device}
cfg.value_preprocessor = RunningStandardScaler
cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
cfg.grad_norm_clip = 1.0
cfg.value_clip = 0.2
cfg.value_loss_scale = 2.0
cfg.damping = 0.1
cfg.max_kl_divergence = 0.01
cfg.conjugate_gradient_steps = 10
cfg.max_backtrack_steps = 10
cfg.accept_ratio = 0.5
cfg.step_fraction = 1.0
cfg.time_limit_bootstrap = False
cfg.fixed_order = False
# logging to TensorBoard and write checkpoints (in timesteps)
cfg.experiment.write_interval = "auto" if not args.eval else 0
cfg.experiment.checkpoint_interval = "auto" if not args.eval else 0
cfg.experiment.directory = f"runs/torch/{task_name}"

agent = HATRPO(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,
    cfg=cfg,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 4800, "headless": not args.visualizer, "environment_info": "log"}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
