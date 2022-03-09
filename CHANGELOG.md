# Changelog

Record of notable changes in this project

## [Unreleased]

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

First official release
