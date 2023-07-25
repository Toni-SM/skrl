# [start-tensorboard-configuration]
DEFAULT_CONFIG = {
    # ...

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-tensorboard-configuration]


# [start-wandb-configuration]
DEFAULT_CONFIG = {
    # ...

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-wandb-configuration]


# [start-checkpoint-configuration]
DEFAULT_CONFIG = {
    # ...

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-checkpoint-configuration]


# [start-checkpoint-load-agent-torch]
from skrl.agents.torch.ppo import PPO

# Instantiate the agent
agent = PPO(models=models,  # models dict
            memory=memory,  # memory instance, or None if not required
            cfg=agent_cfg,  # configuration dict (preprocessors, learning rate schedulers, etc.)
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)

# Load the checkpoint
agent.load("./runs/22-09-29_22-48-49-816281_DDPG/checkpoints/agent_1200.pt")
# [end-checkpoint-load-agent-torch]


# [start-checkpoint-load-agent-jax]
from skrl.agents.jax.ppo import PPO

# Instantiate the agent
agent = PPO(models=models,  # models dict
            memory=memory,  # memory instance, or None if not required
            cfg=agent_cfg,  # configuration dict (preprocessors, learning rate schedulers, etc.)
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)

# Load the checkpoint
agent.load("./runs/22-09-29_22-48-49-816281_DDPG/checkpoints/agent_1200.pickle")
# [end-checkpoint-load-agent-jax]


# [start-checkpoint-load-model-torch]
from skrl.models.torch import Model, DeterministicMixin

# Define the model
class Policy(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# Instantiate the model
policy = Policy(env.observation_space, env.action_space, env.device, clip_actions=True)

# Load the checkpoint
policy.load("./runs/22-09-29_22-48-49-816281_DDPG/checkpoints/2500_policy.pt")
# [end-checkpoint-load-model-torch]


# [start-checkpoint-load-model-jax]
from skrl.models.jax import Model, DeterministicMixin

# Define the model
class Policy(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device=None, clip_actions=False, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.Dense(32)(inputs["states"])
        x = nn.relu(x)
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        return x, {}

# Instantiate the model
policy = Policy(env.observation_space, env.action_space, env.device, clip_actions=True)

# Load the checkpoint
policy.load("./runs/22-09-29_22-48-49-816281_DDPG/checkpoints/2500_policy.pickle")
# [end-checkpoint-load-model-jax]


# [start-checkpoint-load-huggingface-torch]
from skrl.agents.torch.ppo import PPO
from skrl.utils.huggingface import download_model_from_huggingface

# Instantiate the agent
agent = PPO(models=models,  # models dict
            memory=memory,  # memory instance, or None if not required
            cfg=agent_cfg,  # configuration dict (preprocessors, learning rate schedulers, etc.)
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)

# Load the checkpoint from Hugging Face Hub
path = download_model_from_huggingface("skrl/OmniIsaacGymEnvs-Cartpole-PPO", filename="agent.pt")
agent.load(path)
# [end-checkpoint-load-huggingface-torch]


# [start-checkpoint-load-huggingface-jax]
from skrl.agents.jax.ppo import PPO
from skrl.utils.huggingface import download_model_from_huggingface

# Instantiate the agent
agent = PPO(models=models,  # models dict
            memory=memory,  # memory instance, or None if not required
            cfg=agent_cfg,  # configuration dict (preprocessors, learning rate schedulers, etc.)
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)

# Load the checkpoint from Hugging Face Hub
path = download_model_from_huggingface("skrl/OmniIsaacGymEnvs-Cartpole-PPO", filename="agent.pickle")
agent.load(path)
# [end-checkpoint-load-huggingface-jax]


# [start-checkpoint-migrate-agent-torch]
from skrl.agents.torch.ppo import PPO

# Instantiate the agent
agent = PPO(models=models,  # models dict
            memory=memory,  # memory instance, or None if not required
            cfg=agent_cfg,  # configuration dict (preprocessors, learning rate schedulers, etc.)
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)

# Migrate a rl_games checkpoint
agent.migrate(path="./runs/Cartpole/nn/Cartpole.pth")
# [end-checkpoint-migrate-agent-torch]


# [start-checkpoint-migrate-model-torch]
from skrl.models.torch import Model, DeterministicMixin

# Define the model
class Policy(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}

# Instantiate the model
policy = Policy(env.observation_space, env.action_space, env.device, clip_actions=True)

# Migrate a rl_games checkpoint (only the model)
policy.migrate(path="./runs/Cartpole/nn/Cartpole.pth")

# or migrate a stable-baselines3 checkpoint
policy.migrate(path="./ddpg_pendulum.zip")

# or migrate a checkpoint of any other library
state_dict = torch.load("./external_model.pt")
policy.migrate(state_dict=state_dict)
# [end-checkpoint-migrate-model-torch]


# [start-export-memory-torch]
from skrl.memories.torch import RandomMemory

# Instantiate a memory and enable its export
memory = RandomMemory(memory_size=16,
                      num_envs=env.num_envs,
                      device=device,
                      export=True,
                      export_format="pt",
                      export_directory="./memories")
# [end-export-memory-torch]


# [start-export-memory-jax]
from skrl.memories.jax import RandomMemory

# Instantiate a memory and enable its export
memory = RandomMemory(memory_size=16,
                      num_envs=env.num_envs,
                      device=device,
                      export=True,
                      export_format="np",
                      export_directory="./memories")
# [end-export-memory-jax]
