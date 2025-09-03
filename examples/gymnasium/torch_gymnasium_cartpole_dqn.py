import argparse
import os
import gymnasium as gym

# import the skrl components to build the RL system
from skrl import logger
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.torch import deterministic_model


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


# load the environment (note: the environment version may change depending on the gymnasium version)
task_name = "CartPole"
render_mode = "human" if not args.headless else None
env_id = [spec for spec in gym.envs.registry if spec.startswith(f"{task_name}-v")][-1]  # get latest environment version
if args.num_envs <= 1:
    env = gym.make(env_id, render_mode=render_mode)
else:
    env = gym.make_vec(env_id, num_envs=args.num_envs, render_mode=render_mode, vectorization_mode="sync")
# wrap the environment
env = wrap_env(env)

device = env.device


# instantiate a memory as experience replay
memory = RandomMemory(memory_size=50000, num_envs=env.num_envs, device=device, replacement=False)


# instantiate the agent's models (function approximators) using model instantiators.
# DQN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#models
network = [{"name": "net", "input": "OBSERVATIONS", "layers": [64, 64], "activations": "relu"}]
models = {}
models["q_network"] = deterministic_model(
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
    network=network,
    output="ACTIONS",
)
models["target_q_network"] = deterministic_model(
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
    network=network,
    output="ACTIONS",
)

# initialize models' lazy modules
for role, model in models.items():
    model.init_state_dict(role=role)

# initialize models' parameters (weights and biases)
for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/dqn.html#configuration-and-hyperparameters
cfg = DQN_DEFAULT_CONFIG.copy()
cfg["learning_starts"] = 100
cfg["exploration"]["final_epsilon"] = 0.04
cfg["exploration"]["timesteps"] = 1500
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = "auto" if not args.eval else 0
cfg["experiment"]["checkpoint_interval"] = "auto" if not args.eval else 0
cfg["experiment"]["directory"] = f"runs/torch/{task_name}"

agent = DQN(
    models=models,
    memory=memory,
    cfg=cfg,
    observation_space=env.observation_space,
    state_space=env.state_space,
    action_space=env.action_space,
    device=device,
)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50000, "headless": args.headless}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

if args.checkpoint:
    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint file not found: '{args.checkpoint}'")
        exit(1)
    agent.load(args.checkpoint)

trainer.train() if not args.eval else trainer.eval()
