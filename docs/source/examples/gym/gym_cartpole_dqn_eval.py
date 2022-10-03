import gym

# Import the skrl components to build the RL system
from skrl.utils.model_instantiators import deterministic_model, Shape
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env


# Load and wrap the Gym environment.
# Note: the environment version may change depending on the gym version
try:
    env = gym.make("CartPole-v0")
except gym.error.DeprecatedEnv as e:
    env_id = [spec.id for spec in gym.envs.registry.all() if spec.id.startswith("CartPole-v")][0]
    print("CartPole-v0 not found. Trying {}".format(env_id))
    env = gym.make(env_id)
env = wrap_env(env)

device = env.device


# Instantiate only the policy for evaluation.
# DQN requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.dqn.html#spaces-and-models
models_dqn = {}
models_dqn["q_network"] = deterministic_model(observation_space=env.observation_space, 
                                              action_space=env.action_space,
                                              device=device,
                                              clip_actions=False, 
                                              input_shape=Shape.OBSERVATIONS,
                                              hiddens=[64, 64],
                                              hidden_activation=["relu", "relu"],
                                              output_shape=Shape.ACTIONS,
                                              output_activation=None,
                                              output_scale=1.0)


# Configure and instantiate the agent.
# Only modify some of the default configuration, visit its documentation to see all the options
# https://skrl.readthedocs.io/en/latest/modules/skrl.agents.dqn.html#configuration-and-hyperparameters
cfg_dqn = DQN_DEFAULT_CONFIG.copy()
cfg_dqn["exploration"]["timesteps"] = 0
# # logging to TensorBoard each 1000 timesteps and ignore checkpoints
cfg_dqn["experiment"]["write_interval"] = 1000
cfg_dqn["experiment"]["checkpoint_interval"] = 0

agent_dqn = DQN(models=models_dqn, 
                memory=None, 
                cfg=cfg_dqn, 
                observation_space=env.observation_space, 
                action_space=env.action_space,
                device=device)

# load checkpoint
agent_dqn.load("./runs/22-09-10_10-48-10-551426_DQN/checkpoints/best_agent.pt")


# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 50000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent_dqn)

# evaluate the agent
trainer.eval()
