import argparse
import os
import gym

import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.sarsa import SARSA, SARSA_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import tabular_model


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


# load the environment (note: the environment version may change depending on the gym version)
task_name = "Taxi"
render_mode = "human" if not args.headless else None
env_id = [spec for spec in gym.envs.registry if spec.startswith(f"{task_name}-v")][-1]  # get latest environment version
if args.num_envs <= 1:
    env = gym.make(env_id, render_mode=render_mode)
else:
    env = gym.vector.make(env_id, num_envs=args.num_envs, render_mode=render_mode, asynchronous=False)
# wrap the environment
env = wrap_env(env)

device = env.device


# instantiate the agent's model (table)
# SARSA requires 1 model, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/sarsa.html#models
models = {}
models["policy"] = tabular_model(
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
    variant="epsilon-greedy",
    variant_kwargs={"epsilon": 0.1},
)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/sarsa.html#configuration-and-hyperparameters
cfg = SARSA_DEFAULT_CONFIG.copy()
cfg["discount_factor"] = 0.999
cfg["alpha"] = 0.4
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = "auto" if not args.eval else 0
cfg["experiment"]["checkpoint_interval"] = "auto" if not args.eval else 0
cfg["experiment"]["directory"] = f"runs/torch/{task_name}"

agent = SARSA(
    models=models,
    memory=None,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 80000, "headless": args.headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
