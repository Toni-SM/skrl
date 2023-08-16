# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2023-08-16

Transition from pre-release versions (`1.0.0-rc.1` and`1.0.0-rc.2`) to a stable version.

This release also announces the publication of the **skrl** paper in the Journal of Machine Learning Research (JMLR): https://www.jmlr.org/papers/v24/23-0112.html

Summary of the most relevant features:
- JAX support
- New documentation theme and structure
- Multi-agent Reinforcement Learning (MARL)

## [1.0.0-rc.2] - 2023-08-11
### Added
- Get truncation from `time_outs` info in Isaac Gym, Isaac Orbit and Omniverse Isaac Gym environments
- Time-limit (truncation) boostrapping in on-policy actor-critic agents
- Model instantiators `initial_log_std` parameter to set the log standard deviation's initial value

### Changed (breaking changes)
- Structure environment loaders and wrappers file hierarchy coherently
  Import statements now follow the next convention:
  - Wrappers (e.g.):
    - `from skrl.envs.wrappers.torch import wrap_env`
    - `from skrl.envs.wrappers.jax import wrap_env`
  - Loaders (e.g.):
    - `from skrl.envs.loaders.torch import load_omniverse_isaacgym_env`
    - `from skrl.envs.loaders.jax import load_omniverse_isaacgym_env`

### Changed
- Drop support for versions prior to PyTorch 1.9 (1.8.0 and 1.8.1)

## [1.0.0-rc.1] - 2023-07-25
### Added
- JAX support (with Flax and Optax)
- RPO agent
- IPPO and MAPPO multi-agent
- Multi-agent base class
- Bi-DexHands environment loader
- Wrapper for PettingZoo and Bi-DexHands environments
- Parameters `num_envs`, `headless` and `cli_args` for configuring Isaac Gym, Isaac Orbit
and Omniverse Isaac Gym environments when they are loaded

### Changed
- Migrate to `pyproject.toml` Python package development
- Define ML framework dependencies as optional dependencies in the library installer
- Move agent implementations with recurrent models to a separate file
- Allow closing the environment at the end of execution instead of after training/evaluation
- Documentation theme from *sphinx_rtd_theme* to *furo*
- Update documentation structure and examples

### Fixed
- Compatibility for Isaac Sim or OmniIsaacGymEnvs (2022.2.0 or earlier)
- Disable PyTorch gradient computation during the environment stepping
- Get categorical models' entropy
- Typo in `KLAdaptiveLR` learning rate scheduler
  (keep the old name for compatibility with the examples of previous versions.
  The old name will be removed in future releases)

## [0.10.2] - 2023-03-23
### Changed
- Update loader and utils for OmniIsaacGymEnvs 2022.2.1.0
- Update Omniverse Isaac Gym real-world examples

## [0.10.1] - 2023-01-26
### Fixed
- Tensorboard writer instantiation when `write_interval` is zero

## [0.10.0] - 2023-01-22
### Added
- Isaac Orbit environment loader
- Wrap an Isaac Orbit environment
- Gaussian-Deterministic shared model instantiator

## [0.9.1] - 2023-01-17
### Added
- Utility for downloading models from Hugging Face Hub

### Fixed
- Initialization of agent components if they have not been defined
- Manual trainer `train`/`eval` method default arguments

## [0.9.0] - 2023-01-13
### Added
- Support for Farama Gymnasium interface
- Wrapper for robosuite environments
- Weights & Biases integration
- Set the running mode (training or evaluation) of the agents
- Allow clipping the gradient norm for DDPG, TD3 and SAC agents
- Initialize model biases
- Add RNN (RNN, LSTM, GRU and any other variant) support for A2C, DDPG, PPO, SAC, TD3 and TRPO agents
- Allow disabling training/evaluation progressbar
- Farama Shimmy and robosuite examples
- KUKA LBR iiwa real-world example

### Changed (breaking changes)
- Forward model inputs as a Python dictionary
- Returns a Python dictionary with extra output values in model calls

### Changed
- Adopt the implementation of `terminated` and `truncated` over `done` for all environments

### Fixed
- Omniverse Isaac Gym simulation speed for the Franka Emika real-world example
- Call agents' method `record_transition` instead of parent method
to allow storing samples in memories during evaluation
- Move TRPO policy optimization out of the value optimization loop
- Access to the categorical model distribution
- Call reset only once for Gym/Gymnasium vectorized environments

### Removed
- Deprecated method `start` in trainers

## [0.8.0] - 2022-10-03
### Added
- AMP agent for physics-based character animation
- Manual trainer
- Gaussian model mixin
- Support for creating shared models
- Parameter `role` to model methods
- Wrapper compatibility with the new OpenAI Gym environment API
- Internal library colored logger
- Migrate checkpoints/models from other RL libraries to skrl models/agents
- Configuration parameter `store_separately` to agent configuration dict
- Save/load agent modules (models, optimizers, preprocessors)
- Set random seed and configure deterministic behavior for reproducibility
- Benchmark results for Isaac Gym and Omniverse Isaac Gym on the GitHub discussion page
- Franka Emika real-world example

### Changed (breaking changes)
- Models implementation as Python mixin

### Changed
- Multivariate Gaussian model (`GaussianModel` until 0.7.0) to `MultivariateGaussianMixin`
- Trainer's `cfg` parameter position and default values
- Show training/evaluation display progress using `tqdm`
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
