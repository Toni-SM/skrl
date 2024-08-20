# [start-runner-train-torch]
from skrl.utils.runner.torch import Runner
from skrl.envs.wrappers.torch import wrap_env

# load and wrap some environment
env = ...
env = wrap_env(env)

# load the experiment config and instantiate the runner
cfg = Runner.load_cfg_from_yaml("path/to/cfg.yaml")
runner = Runner(env, cfg)

# load a checkpoint to continue training or for evaluation (optional)
runner.agent.load("path/to/checkpoints/agent.pt")

# run the training
runner.run("train")  # or "eval" for evaluation
# [end-runner-train-torch]


# [start-runner-train-jax]
from skrl.utils.runner.jax import Runner
from skrl.envs.wrappers.jax import wrap_env

# load and wrap some environment
env = ...
env = wrap_env(env)

# load the experiment config and instantiate the runner
cfg = Runner.load_cfg_from_yaml("path/to/cfg.yaml")
runner = Runner(env, cfg)

# load a checkpoint to continue training or for evaluation (optional)
runner.agent.load("path/to/checkpoints/agent.pickle")

# run the training
runner.run("train")  # or "eval" for evaluation
# [end-runner-train-jax]

# =============================================================================

# [start-cfg-yaml]
seed: 42

# Models are instantiated using skrl's model instantiator utility
# https://skrl.readthedocs.io/en/latest/api/utils/model_instantiators.html
models:
  separate: False
  policy:  # gaussian model
    class: "GaussianMixin"
    clip_actions: True
    clip_log_std: True
    initial_log_std: 0
    min_log_std: -20.0
    max_log_std: 2.0
    input_shape: "Shape.STATES"
    hiddens: [32, 32]
    hidden_activation: ["elu", "elu"]
    output_shape: "Shape.ACTIONS"
    output_activation: "tanh"
    output_scale: 1.0
  value:  # deterministic model
    class: "DeterministicMixin"
    clip_actions: False
    input_shape: "Shape.STATES"
    hiddens: [32, 32]
    hidden_activation: ["elu", "elu"]
    output_shape: "Shape.ONE"
    output_activation: ""
    output_scale: 1.0

# Memory
# https://skrl.readthedocs.io/en/latest/api/memories/random.html
memory:
  class: "RandomMemory"
  memory_size: -1  # -1: automatically determined value

# PPO agent configuration (field names are from PPO_DEFAULT_CONFIG)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html
agent:
  class: "PPO"
  rollouts: 16
  learning_epochs: 8
  mini_batches: 1
  discount_factor: 0.99
  lambda: 0.95
  learning_rate: 3.e-4
  learning_rate_scheduler: "KLAdaptiveLR"
  learning_rate_scheduler_kwargs:
    kl_threshold: 0.008
  state_preprocessor: "RunningStandardScaler"
  state_preprocessor_kwargs: null
  value_preprocessor: "RunningStandardScaler"
  value_preprocessor_kwargs: null
  random_timesteps: 0
  learning_starts: 0
  grad_norm_clip: 1.0
  ratio_clip: 0.2
  value_clip: 0.2
  clip_predicted_values: True
  entropy_loss_scale: 0.0
  value_loss_scale: 2.0
  kl_threshold: 0
  rewards_shaper_scale: 1.0
  # logging and checkpoint
  experiment:
    directory: "runs"
    experiment_name: ""
    write_interval: 16
    checkpoint_interval: 80

# Sequential trainer
# https://skrl.readthedocs.io/en/latest/api/trainers/sequential.html
trainer:
  class: "SequentialTrainer"
  timesteps: 1600
  environment_info: "log"
# [end-cfg-yaml]
