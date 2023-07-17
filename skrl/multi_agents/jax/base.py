from typing import Any, Mapping, Optional, Sequence, Union

import collections
import copy
import datetime
import os
import pickle
import gym
import gymnasium

import flax
import jax
import numpy as np

from skrl import config, logger
from skrl.memories.jax import Memory
from skrl.models.jax import Model


class MultiAgent:
    def __init__(self,
                 possible_agents: Sequence[str],
                 models: Mapping[str, Mapping[str, Model]],
                 memories: Optional[Mapping[str, Memory]] = None,
                 observation_spaces: Optional[Mapping[str, Union[int, Sequence[int], gym.Space, gymnasium.Space]]] = None,
                 action_spaces: Optional[Mapping[str, Union[int, Sequence[int], gym.Space, gymnasium.Space]]] = None,
                 device: Optional[Union[str, jax.Device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Base class that represent a RL multi-agent

        :param possible_agents: Name of all possible agents the environment could generate
        :type possible_agents: list of str
        :param models: Models used by the agents.
                       External keys are environment agents' names. Internal keys are the models required by the algorithm
        :type models: nested dictionary of skrl.models.jax.Model
        :param memories: Memories to storage the transitions.
        :type memories: dictionary of skrl.memory.jax.Memory, optional
        :param observation_spaces: Observation/state spaces or shapes (default: ``None``)
        :type observation_spaces: dictionary of int, sequence of int, gym.Space or gymnasium.Space, optional
        :param action_spaces: Action spaces or shapes (default: ``None``)
        :type action_spaces: dictionary of int, sequence of int, gym.Space or gymnasium.Space, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or jax.Device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        """
        self._jax = config.jax.backend == "jax"

        self.possible_agents = possible_agents
        self.num_agents = len(self.possible_agents)

        self.models = models
        self.memories = memories
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces

        self.cfg = cfg if cfg is not None else {}

        if device is None:
            self.device = jax.devices()[0]
        else:
            self.device = device if isinstance(device, jax.Device) else jax.devices(device)[0]

        # convert the models to their respective device
        for _models in self.models.values():
            for model in _models.values():
                if model is not None:
                    pass

        self.tracking_data = collections.defaultdict(list)
        self.write_interval = self.cfg.get("experiment", {}).get("write_interval", 1000)

        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        self._cumulative_rewards = None
        self._cumulative_timesteps = None

        self.training = True

        # checkpoint
        self.checkpoint_modules = {uid: {} for uid in self.possible_agents}
        self.checkpoint_interval = self.cfg.get("experiment", {}).get("checkpoint_interval", 1000)
        self.checkpoint_store_separately = self.cfg.get("experiment", {}).get("store_separately", False)
        self.checkpoint_best_modules = {"timestep": 0, "reward": -2 ** 31, "saved": True, "modules": {}}

        # experiment directory
        directory = self.cfg.get("experiment", {}).get("directory", "")
        experiment_name = self.cfg.get("experiment", {}).get("experiment_name", "")
        if not directory:
            directory = os.path.join(os.getcwd(), "runs")
        if not experiment_name:
            experiment_name = f"{datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S-%f')}_{self.__class__.__name__}"
        self.experiment_dir = os.path.join(directory, experiment_name)

    def __str__(self) -> str:
        """Generate a representation of the agent as string

        :return: Representation of the agent as string
        :rtype: str
        """
        string = f"Multi-agent: {repr(self)}"
        for k, v in self.cfg.items():
            if type(v) is dict:
                string += f"\n  |-- {k}"
                for k1, v1 in v.items():
                    string += f"\n  |     |-- {k1}: {v1}"
            else:
                string += f"\n  |-- {k}: {v}"
        return string

    def _as_dict(self, _input: Any) -> Mapping[str, Any]:
        """Convert a configuration value into a dictionary according to the number of agents

        :param _input: Configuration value
        :type _input: Any

        :raises ValueError: The configuration value is a dictionary different from the number of agents

        :return: Configuration value as a dictionary
        :rtype: list of any configuration value
        """
        if _input and isinstance(_input, collections.abc.Mapping):
            if set(_input) < set(self.possible_agents):
                logger.error("The configuration value does not match possible agents")
                raise ValueError("The configuration value does not match possible agents")
            elif set(_input) >= set(self.possible_agents):
                return _input
        try:
            return {name: copy.deepcopy(_input) for name in self.possible_agents}
        except TypeError:
            return {name: _input for name in self.possible_agents}

    def _empty_preprocessor(self, _input: Any, *args, **kwargs) -> Any:
        """Empty preprocess method

        This method is defined because PyTorch multiprocessing can't pickle lambdas

        :param _input: Input to preprocess
        :type _input: Any

        :return: Preprocessed input
        :rtype: Any
        """
        return _input

    def _get_internal_value(self, _module: Any) -> Any:
        """Get internal module/variable state/value

        :param _module: Module or variable
        :type _module: Any

        :return: Module/variable state/value
        :rtype: Any
        """
        return _module.state_dict.params if hasattr(_module, "state_dict") else _module

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent

        This method should be called before the agent is used.
        It will initialize the TensoBoard writer (and optionally Weights & Biases) and create the checkpoints directory

        :param trainer_cfg: Trainer configuration
        :type trainer_cfg: dict, optional
        """
        # setup Weights & Biases
        if self.cfg.get("experiment", {}).get("wandb", False):
            # save experiment config
            trainer_cfg = trainer_cfg if trainer_cfg is not None else {}
            try:
                models_cfg = {uid: {k: v.net._modules for (k, v) in self.models[uid].items()} for uid in self.possible_agents}
            except AttributeError:
                models_cfg = {uid: {k: v._modules for (k, v) in self.models[uid].items()} for uid in self.possible_agents}
            config={**self.cfg, **trainer_cfg, **models_cfg}
            # set default values
            wandb_kwargs = copy.deepcopy(self.cfg.get("experiment", {}).get("wandb_kwargs", {}))
            wandb_kwargs.setdefault("name", os.path.split(self.experiment_dir)[-1])
            wandb_kwargs.setdefault("sync_tensorboard", True)
            wandb_kwargs.setdefault("config", {})
            wandb_kwargs["config"].update(config)
            # init Weights & Biases
            import wandb
            wandb.init(**wandb_kwargs)

        # main entry to log data for consumption and visualization by TensorBoard
        if self.write_interval > 0:
            self.writer = None
            # tensorboard via torch SummaryWriter
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.experiment_dir)
            except ImportError as e:
                pass
            # tensorboard via tensorflow
            if self.writer is None:
                try:
                    import tensorflow

                    class _SummaryWriter:
                        def __init__(self, log_dir):
                            self.writer = tensorflow.summary.create_file_writer(logdir=log_dir)

                        def add_scalar(self, tag, value, step):
                            with self.writer.as_default():
                                tensorflow.summary.scalar(tag, value, step=step)

                    self.writer = _SummaryWriter(log_dir=self.experiment_dir)
                except ImportError as e:
                    pass
            # tensorboard via tensorboardX
            if self.writer is None:
                try:
                    import tensorboardX
                    self.writer = tensorboardX.SummaryWriter(log_dir=self.experiment_dir)
                except ImportError as e:
                    pass
            # show warnings and exit
            if self.writer is None:
                logger.warning("No package found to write events to Tensorboard.")
                logger.warning("Set agent's `write_interval` setting to 0 to disable writing")
                logger.warning("or install one of the following packages:")
                logger.warning("  - PyTorch: https://pytorch.org/get-started/locally")
                logger.warning("  - TensorFlow: https://www.tensorflow.org/install")
                logger.warning("  - TensorboardX: https://github.com/lanpa/tensorboardX#install")
                logger.warning("The current running process will be terminated.")
                exit()

        if self.checkpoint_interval > 0:
            os.makedirs(os.path.join(self.experiment_dir, "checkpoints"), exist_ok=True)

    def track_data(self, tag: str, value: float) -> None:
        """Track data to TensorBoard

        Currently only scalar data are supported

        :param tag: Data identifier (e.g. 'Loss / policy loss')
        :type tag: str
        :param value: Value to track
        :type value: float
        """
        self.tracking_data[tag].append(value)

    def write_tracking_data(self, timestep: int, timesteps: int) -> None:
        """Write tracking data to TensorBoard

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        for k, v in self.tracking_data.items():
            if k.endswith("(min)"):
                self.writer.add_scalar(k, np.min(v), timestep)
            elif k.endswith("(max)"):
                self.writer.add_scalar(k, np.max(v), timestep)
            else:
                self.writer.add_scalar(k, np.mean(v), timestep)
        # reset data containers for next iteration
        self._track_rewards.clear()
        self._track_timesteps.clear()
        self.tracking_data.clear()

    def write_checkpoint(self, timestep: int, timesteps: int) -> None:
        """Write checkpoint (modules) to disk

        The checkpoints are saved in the directory 'checkpoints' in the experiment directory.
        The name of the checkpoint is the current timestep if timestep is not None, otherwise it is the current time.

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        tag = str(timestep if timestep is not None else datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f"))
        # separated modules
        if self.checkpoint_store_separately:
            for uid in self.possible_agents:
                for name, module in self.checkpoint_modules[uid].items():
                    with open(os.path.join(self.experiment_dir, "checkpoints", f"{uid}_{name}_{tag}.pickle"), "wb") as file:
                        pickle.dump(flax.serialization.to_bytes(self._get_internal_value(module)), file, protocol=4)
        # whole agent
        else:
            modules = {uid: {name: flax.serialization.to_bytes(self._get_internal_value(module)) for name, module in self.checkpoint_modules[uid].items()} \
                       for uid in self.possible_agents}

            with open(os.path.join(self.experiment_dir, "checkpoints", f"agent_{tag}.pickle"), "wb") as file:
                pickle.dump(modules, file, protocol=4)

        # best modules
        if self.checkpoint_best_modules["modules"] and not self.checkpoint_best_modules["saved"]:
            # separated modules
            if self.checkpoint_store_separately:
                for uid in self.possible_agents:
                    for name, module in self.checkpoint_modules.items():
                        with open(os.path.join(self.experiment_dir, "checkpoints", f"best_{uid}_{name}.pickle"), "wb") as file:
                            pickle.dump(flax.serialization.to_bytes(self.checkpoint_best_modules["modules"][uid][name]), file, protocol=4)
            # whole agent
            else:
                modules = {uid: {name: flax.serialization.to_bytes(self.checkpoint_best_modules["modules"][uid][name]) \
                                 for name in self.checkpoint_modules[uid].keys()} for uid in self.possible_agents}
                with open(os.path.join(self.experiment_dir, "checkpoints", "best_agent.pickle"), "wb") as file:
                    pickle.dump(modules, file, protocol=4)
            self.checkpoint_best_modules["saved"] = True

    def act(self, states: Mapping[str, Union[np.ndarray, jax.Array]], timestep: int, timesteps: int) -> Union[np.ndarray, jax.Array]:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: dictionary of np.ndarray or jax.Array
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :raises NotImplementedError: The method is not implemented by the inheriting classes

        :return: Actions
        :rtype: np.ndarray or jax.Array
        """
        raise NotImplementedError

    def record_transition(self,
                          states: Mapping[str, Union[np.ndarray, jax.Array]],
                          actions: Mapping[str, Union[np.ndarray, jax.Array]],
                          rewards: Mapping[str, Union[np.ndarray, jax.Array]],
                          next_states: Mapping[str, Union[np.ndarray, jax.Array]],
                          terminated: Mapping[str, Union[np.ndarray, jax.Array]],
                          truncated: Mapping[str, Union[np.ndarray, jax.Array]],
                          infos: Mapping[str, Any],
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory (to be implemented by the inheriting classes)

        Inheriting classes must call this method to record episode information (rewards, timesteps, etc.).
        In addition to recording environment transition (such as states, rewards, etc.), agent information can be recorded.

        :param states: Observations/states of the environment used to make the decision
        :type states: dictionary of np.ndarray or jax.Array
        :param actions: Actions taken by the agent
        :type actions: dictionary of np.ndarray or jax.Array
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: dictionary of np.ndarray or jax.Array
        :param next_states: Next observations/states of the environment
        :type next_states: dictionary of np.ndarray or jax.Array
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: dictionary of np.ndarray or jax.Array
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: dictionary of np.ndarray or jax.Array
        :param infos: Additional information about the environment
        :type infos: dictionary of any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if self.write_interval > 0:
            _rewards = next(iter(rewards.values()))
            _terminated = next(iter(terminated.values()))
            _truncated = next(iter(truncated.values()))

            # compute the cumulative sum of the rewards and timesteps
            if self._cumulative_rewards is None:
                self._cumulative_rewards = np.zeros_like(_rewards, dtype=np.float32)
                self._cumulative_timesteps = np.zeros_like(_rewards, dtype=np.int32)

            # TODO: find a better way to avoid https://jax.readthedocs.io/en/latest/errors.html#jax.errors.ConcretizationTypeError
            if self._jax:
                _rewards = jax.device_get(_rewards)
                _terminated = jax.device_get(_terminated)
                _truncated = jax.device_get(_truncated)

            self._cumulative_rewards += _rewards
            self._cumulative_timesteps += 1

            # check ended episodes
            finished_episodes = (_terminated + _truncated).nonzero()[0]
            if finished_episodes.size:

                # storage cumulative rewards and timesteps
                self._track_rewards.extend(self._cumulative_rewards[finished_episodes][:, 0].reshape(-1).tolist())
                self._track_timesteps.extend(self._cumulative_timesteps[finished_episodes][:, 0].reshape(-1).tolist())

                # reset the cumulative rewards and timesteps
                self._cumulative_rewards[finished_episodes] = 0
                self._cumulative_timesteps[finished_episodes] = 0

            # record data
            self.tracking_data["Reward / Instantaneous reward (max)"].append(np.max(_rewards).item())
            self.tracking_data["Reward / Instantaneous reward (min)"].append(np.min(_rewards).item())
            self.tracking_data["Reward / Instantaneous reward (mean)"].append(np.mean(_rewards).item())

            if len(self._track_rewards):
                track_rewards = np.array(self._track_rewards)
                track_timesteps = np.array(self._track_timesteps)

                self.tracking_data["Reward / Total reward (max)"].append(np.max(track_rewards))
                self.tracking_data["Reward / Total reward (min)"].append(np.min(track_rewards))
                self.tracking_data["Reward / Total reward (mean)"].append(np.mean(track_rewards))

                self.tracking_data["Episode / Total timesteps (max)"].append(np.max(track_timesteps))
                self.tracking_data["Episode / Total timesteps (min)"].append(np.min(track_timesteps))
                self.tracking_data["Episode / Total timesteps (mean)"].append(np.mean(track_timesteps))

    def set_mode(self, mode: str) -> None:
        """Set the model mode (training or evaluation)

        :param mode: Mode: 'train' for training or 'eval' for evaluation
        :type mode: str
        """
        for _models in self.models.values():
            for model in _models.values():
                if model is not None:
                    model.set_mode(mode)

    def set_running_mode(self, mode: str) -> None:
        """Set the current running mode (training or evaluation)

        This method sets the value of the ``training`` property (boolean).
        This property can be used to know if the agent is running in training or evaluation mode.

        :param mode: Mode: 'train' for training or 'eval' for evaluation
        :type mode: str
        """
        self.training = mode == "train"

    def save(self, path: str) -> None:
        """Save the agent to the specified path

        :param path: Path to save the model to
        :type path: str
        """
        modules = {uid: {name: flax.serialization.to_bytes(self._get_internal_value(module)) \
                         for name, module in self.checkpoint_modules[uid].items()} for uid in self.possible_agents}

        # HACK: Does it make sense to use https://github.com/google/orbax
        # file.write(flax.serialization.to_bytes(modules))
        with open(path, "wb") as file:
            pickle.dump(modules, file, protocol=4)

    def load(self, path: str) -> None:
        """Load the model from the specified path

        :param path: Path to load the model from
        :type path: str
        """
        with open(path, "rb") as file:
            modules = pickle.load(file)
        if type(modules) is dict:
            for uid in self.possible_agents:
                if uid not in modules:
                    logger.warning(f"Cannot load modules for {uid}. The agent doesn't have such an instance")
                    continue
                for name, data in modules[uid].items():
                    module = self.checkpoint_modules[uid].get(name, None)
                    if module is not None:
                        if hasattr(module, "load_state_dict"):
                            params = flax.serialization.from_bytes(module.state_dict.params, data)
                            module.state_dict = module.state_dict.replace(params=params)
                        else:
                            pass  # TODO:raise NotImplementedError
                    else:
                        logger.warning(f"Cannot load the {uid}:{name} module. The agent doesn't have such an instance")

    def migrate(self,
                path: str,
                name_map: Mapping[str, Mapping[str, str]] = {},
                auto_mapping: bool = True,
                verbose: bool = False) -> bool:
        """Migrate the specified extrernal checkpoint to the current agent

        :raises NotImplementedError: Not yet implemented
        """
        raise NotImplementedError

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        timestep += 1

        # update best models and write checkpoints
        if timestep > 1 and self.checkpoint_interval > 0 and not timestep % self.checkpoint_interval:
            # update best models
            reward = np.mean(self.tracking_data.get("Reward / Total reward (mean)", -2 ** 31))
            if reward > self.checkpoint_best_modules["reward"]:
                self.checkpoint_best_modules["timestep"] = timestep
                self.checkpoint_best_modules["reward"] = reward
                self.checkpoint_best_modules["saved"] = False
                self.checkpoint_best_modules["modules"] = {uid: {k: copy.deepcopy(self._get_internal_value(v)) \
                    for k, v in self.checkpoint_modules[uid].items()} for uid in self.possible_agents}
            # write checkpoints
            self.write_checkpoint(timestep, timesteps)

        # write to tensorboard
        if timestep > 1 and self.write_interval > 0 and not timestep % self.write_interval:
            self.write_tracking_data(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :raises NotImplementedError: The method is not implemented by the inheriting classes
        """
        raise NotImplementedError
