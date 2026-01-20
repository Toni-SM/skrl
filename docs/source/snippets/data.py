from skrl.agents.torch import ExperimentCfg
from skrl.agents.torch import Agent
from skrl.memories.torch import RandomMemory

agent = Agent(cfg={})


# [start-data-configuration]
ExperimentCfg(
    directory="",                # experiment's parent directory
    experiment_name="",          # experiment name
    write_interval="auto",       # TensorBoard writing interval (timesteps)
    checkpoint_interval="auto",  # interval for checkpoints (timesteps)
    store_separately=False,      # whether to store checkpoints separately
    wandb=False,                 # whether to use Weights & Biases
    wandb_kwargs={},             # Weights & Biases arguments
)
# [end-data-configuration]


# [start-data-agent-track-data]
# assuming agent is an instance of an Agent subclass
agent.track_data("Resource / CPU usage", psutil.cpu_percent())
# [end-data-agent-track-data]


# [start-data-agent-writer-add-scalar]
# assuming agent is an instance of an Agent subclass
agent.writer.add_scalar(tag="Resource / CPU usage", value=psutil.cpu_percent(), timestep=1000)
# [end-data-agent-writer-add-scalar]

# =============================================================================

# [start-checkpoint-load-agent]
# assuming agent is an instance of an Agent subclass
agent.load("./runs/22-09-29_22-48-49-816281_DDPG/checkpoints/agent_1200.pt")
# [end-checkpoint-load-agent]


# [start-checkpoint-load-model]
# assuming policy is an instance of a Model subclass
# - policy = StochasticModel(...), or
# - policy = agent.models["policy"] or agent.policy
policy.load("./runs/22-09-29_22-48-49-816281_DDPG/checkpoints/2500_policy.pt")
# [end-checkpoint-load-model]


# [start-checkpoint-load-huggingface]
from skrl.utils.huggingface import download_model_from_huggingface

# assuming agent is an instance of an Agent subclass
path = download_model_from_huggingface("skrl/OmniIsaacGymEnvs-Cartpole-PPO", filename="agent.pt")
agent.load(path)
# [end-checkpoint-load-huggingface]

# =============================================================================

# [start-export-memory]
memory = RandomMemory(
    memory_size=16,
    num_envs=env.num_envs,
    device=device,
    export=True,
    export_format="pt",
    export_directory="./memories"
)
# [end-export-memory]
