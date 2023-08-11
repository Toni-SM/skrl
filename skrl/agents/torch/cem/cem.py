from typing import Any, Dict, Optional, Tuple, Union

import copy
import gym
import gymnasium

import torch
import torch.nn.functional as F

from skrl import logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model


# [start-config-dict-torch]
CEM_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "percentile": 0.70,             # percentile to compute the reward bound [0, 1]

    "discount_factor": 0.99,        # discount factor (gamma)

    "learning_rate": 1e-2,          # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

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
# [end-config-dict-torch]


class CEM(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None) -> None:
        """Cross-Entropy Method (CEM)

        https://ieeexplore.ieee.org/abstract/document/6796865/

        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: ``None``)
        :type observation_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param action_space: Action space or shape (default: ``None``)
        :type action_space: int, tuple or list of int, gym.Space, gymnasium.Space or None, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                       If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(CEM_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        # models
        self.policy = self.models.get("policy", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy

        # configuration:
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._percentile = self.cfg["percentile"]
        self._discount_factor = self.cfg["discount_factor"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._episode_tracking = []

        # set up optimizer and learning rate scheduler
        if self.policy is not None:
            self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.int64)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)

        self.tensors_names = ["states", "actions", "rewards", "next_states", "terminated"]

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        states = self._state_preprocessor(states)

        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": states}, role="policy")

        # sample stochastic actions
        return self.policy.act({"states": states}, role="policy")

    def record_transition(self,
                          states: torch.Tensor,
                          actions: torch.Tensor,
                          rewards: torch.Tensor,
                          next_states: torch.Tensor,
                          terminated: torch.Tensor,
                          truncated: torch.Tensor,
                          infos: Any,
                          timestep: int,
                          timesteps: int) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        # reward shaping
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards, timestep, timesteps)

        if self.memory is not None:
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                    terminated=terminated, truncated=truncated)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                   terminated=terminated, truncated=truncated)

        # track episodes internally
        if self._rollout:
            indexes = torch.nonzero(terminated + truncated)
            if indexes.numel():
                for i in indexes[:, 0]:
                    self._episode_tracking[i.item()].append(self._rollout + 1)
        else:
            self._episode_tracking = [[0] for _ in range(rewards.size(-1))]

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
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self._rollout = 0
            self._update(timestep, timesteps)

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        # sample all memory
        sampled_states, sampled_actions, sampled_rewards, _, _ = self.memory.sample_all(names=self.tensors_names)[0]

        sampled_states = self._state_preprocessor(sampled_states, train=True)

        with torch.no_grad():
            # compute discounted return threshold
            limits = []
            returns = []
            for e in range(sampled_rewards.size(-1)):
                for i, j in zip(self._episode_tracking[e][:-1], self._episode_tracking[e][1:]):
                    limits.append([e + i, e + j])
                    rewards = sampled_rewards[e + i: e + j]
                    returns.append(torch.sum(rewards * self._discount_factor ** \
                        torch.arange(rewards.size(0), device=rewards.device).flip(-1).view(rewards.size())))

            if not len(returns):
                logger.warning("No returns to update. Consider increasing the number of rollouts")
                return

            returns = torch.tensor(returns)
            return_threshold = torch.quantile(returns, self._percentile, dim=-1)

            # get elite states and actions
            indexes = torch.nonzero(returns >= return_threshold)
            elite_states = torch.cat([sampled_states[limits[i][0]:limits[i][1]] for i in indexes[:, 0]], dim=0)
            elite_actions = torch.cat([sampled_actions[limits[i][0]:limits[i][1]] for i in indexes[:, 0]], dim=0)

        # compute scores for the elite states
        _, _, outputs = self.policy.act({"states": elite_states}, role="policy")
        scores = outputs["net_output"]

        # compute policy loss
        policy_loss = F.cross_entropy(scores, elite_actions.view(-1))

        # optimization step
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # update learning rate
        if self._learning_rate_scheduler:
            self.scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", policy_loss.item())

        self.track_data("Coefficient / Return threshold", return_threshold.item())
        self.track_data("Coefficient / Mean discounted returns", torch.mean(returns).item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
