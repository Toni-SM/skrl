# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.9.0] - Unreleased
### Added
- Set the running mode (training or evaluation) of the agents
- Weights & Biases integration (by @juhannc)
- Support for Gymnasium interface

### Changed
- Adopt the implementation of `terminated` and `truncated` over `done` for all environments

### Fixed
- Omniverse Isaac Gym simulation speed for the Franka Emika real-world example
- Call agents' method `record_transition` instead of parent method
to allow storing samples in memories during evaluation

### Removed
- Deprecated method `start` in trainers

## [0.8.0] - 2022-10-03
### Added
- AMP agent for physics-based character animation
- Manual trainer
- Gaussian model mixin
- Support for creating shared models
- Parameter `role` to model methods
- Wrapper compatibility with the new OpenAI Gym environment API (by @juhannc)
- Internal library colored logger
- Migrate checkpoints/models from other RL libraries to skrl models/agents
- Configuration parameter `store_separately` to agent configuration dict
- Save/load agent modules (models, optimizers, preprocessors)
- Set random seed and configure deterministic behavior for reproducibility
- Benchmark results for Isaac Gym and Omniverse Isaac Gym on the GitHub discussion page
- Franka Emika real-world example

### Changed
- Models implementation as Python mixin [**breaking change**]
- Multivariate Gaussian model (`GaussianModel` until 0.7.0) to `MultivariateGaussianMixin`
- Trainer's `cfg` parameter position and default values
- Show training/evaluation display progress using `tqdm` (by @juhannc)
- Update Isaac Gym and Omniverse Isaac Gym examples

### Fixed
- Missing recursive arguments during model weights initialization
- Tensor dimension when computing preprocessor parallel variance
- Models' clip tensors dtype to `float32`

### Removed
- Parameter `inference` from model methods
- Configuration parameter `checkpoint_policy_only` from agent configuration dict

## [0.7.0] - 2022-07-11
### Added
- A2C agent
- Isaac Gym (preview 4) environment loader
- Wrap an Isaac Gym (preview 4) environment
- Support for OpenAI Gym vectorized environments
- Running standard scaler for input preprocessing
- Installation from PyPI (`pip install skrl`)

## [0.6.0] - 2022-06-09
### Added
- Omniverse Isaac Gym environment loader
- Wrap an Omniverse Isaac Gym environment
- Save best models during training

## [0.5.0] - 2022-05-18
### Added
- TRPO agent
- DeepMind environment wrapper
- KL Adaptive learning rate scheduler
- Handle `gym.spaces.Dict` observation spaces (OpenAI Gym and DeepMind environments)
- Forward environment info to agent `record_transition` method
- Expose and document the random seeding mechanism
- Define rewards shaping function in agents' config
- Define learning rate scheduler in agents' config
- Improve agent's algorithm description in documentation (PPO and TRPO at the moment)

### Changed
- Compute the Generalized Advantage Estimation (GAE) in agent `_update` method
- Move noises definition to `resources` folder
- Update the Isaac Gym examples

### Removed
- `compute_functions` for computing the GAE from memory base class

## [0.4.1] - 2022-03-22
### Added
- Examples of all Isaac Gym environments (preview 3)
- Tensorboard file iterator for data post-processing

### Fixed
- Init and evaluate agents in ParallelTrainer

## [0.4.0] - 2022-03-09
### Added
- CEM, SARSA and Q-learning agents
- Tabular model
- Parallel training using multiprocessing
- Isaac Gym utilities

### Changed
- Initialize agents in a separate method
- Change the name of the `networks` argument to `models`

### Fixed
- Reset environments after post-processing

## [0.3.0] - 2022-02-07
### Added
- DQN and DDQN agents
- Export memory to files
- Postprocessing utility to iterate over memory files
- Model instantiator utility to allow fast development
- More examples and contents in the documentation

### Fixed
- Clip actions using the whole space's limits

## [0.2.0] - 2022-01-18
### Added
- First official release
