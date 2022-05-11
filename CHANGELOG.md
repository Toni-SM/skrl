# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
### Added
- A2C, and TRPO agents
- DeepMind environment wrapper
- Handle `gym.spaces.Dict` observation spaces (OpenAI Gym and DeepMind Gym environments)
- Forward environment info to agent `record_transition` method
- Expose and document the random seeding mechanism
- Define rewards shaping function in agents' config
- Define learning rate scheduler in agents' config
- Improve agent's algorithm description in documentation

### Changed
- Move GAE (Generalized Advantage Estimation) from memory to agent
- Move noises definition to resources folder
- Update the Isaac Gym examples

## [0.4.1] - 2021-03-22
### Added
- Examples of all Isaac Gym environments (preview 3)
- Tensorboard file iterator for data post-processing

### Fixed
- Init and evaluate agents in ParallelTrainer

## [0.4.0] - 2021-03-09
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

## [0.3.0] - 2021-02-07
### Added
- DQN and DDQN agents
- Export memory to files
- Postprocessing utility to iterate over memory files
- Model instantiator utility to allow fast development
- More examples and contents in the documentation

### Fixed
- Clip actions using the whole space's limits 

## [0.2.0] - 2021-01-18
### Added
- First official release
