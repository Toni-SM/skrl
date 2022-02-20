from typing import Union, Tuple, Dict

import gym
import copy

import torch
import torch.nn.functional as F

from ....memories.torch import Memory
from ....models.torch import Model

from .. import Agent


CEM_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "percentile": 0.70,             # percentile to compute the reward bound [0, 1]

    "discount_factor": 0.99,        # discount factor (gamma)
    
    "learning_rate": 1e-2,          # learning rate

    "random_timesteps": 1000,       # random exploration steps
    "learning_starts": 1000,        # learning starts after this many steps

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "checkpoint_policy_only": True,     # checkpoint for policy only
    }
}


class CEM(Agent):
    def __init__(self, 
                 networks: Dict[str, Model], 
                 memory: Union[Memory, Tuple[Memory], None] = None, 
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 device: Union[str, torch.device] = "cuda:0", 
                 cfg: dict = {}) -> None:
        """Cross-Entropy Method (CEM)

        https://ieeexplore.ieee.org/abstract/document/6796865/
        
        :param networks: Networks used by the agent
        :type networks: dictionary of skrl.models.torch.Model
        :param memory: Memory to storage the transitions.
                       If it is a tuple, the first element will be used for training and 
                       for the rest only the environment transitions will be added
        :type memory: skrl.memory.torch.Memory, list of skrl.memory.torch.Memory or None
        :param observation_space: Observation/state space or shape (default: None)
        :type observation_space: int, tuple or list of integers, gym.Space or None, optional
        :param action_space: Action space or shape (default: None)
        :type action_space: int, tuple or list of integers, gym.Space or None, optional
        :param device: Computing device (default: "cuda:0")
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict

        :raises KeyError: If the networks dictionary is missing a required key
        """
        _cfg = copy.deepcopy(CEM_DEFAULT_CONFIG)
        _cfg.update(cfg)
        super().__init__(networks=networks, 
                         memory=memory, 
                         observation_space=observation_space, 
                         action_space=action_space, 
                         device=device, 
                         cfg=_cfg)

        # networks
        if not "policy" in self.networks.keys():
            raise KeyError("The policy network is not defined under 'policy' key (networks['policy'])")
        
        self.policy = self.networks["policy"]

        # checkpoint networks
        self.checkpoint_networks = self.networks
        
        # configuration:
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._percentile = self.cfg["percentile"]
        self._discount_factor = self.cfg["discount_factor"]
        self._learning_rate = self.cfg["learning_rate"]
        
        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]
        
        self._episode_tracking = []

        # set up optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self._learning_rate)

    def init(self) -> None:
        """Initialize the agent
        """
        super().init()
        
        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.int64)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="dones", size=1, dtype=torch.bool)

        self.tensors_names = ["states", "actions", "rewards", "next_states", "dones"]

    def act(self, 
            states: torch.Tensor, 
            timestep: int, 
            timesteps: int, 
            inference: bool = False) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        :param inference: Flag to indicate whether the network is making inference
        :type inference: bool

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act(states)

        # sample stochastic actions 
        return self.policy.act(states, inference=inference)

    def record_transition(self, 
                          states: torch.Tensor, 
                          actions: torch.Tensor, 
                          rewards: torch.Tensor, 
                          next_states: torch.Tensor, 
                          dones: torch.Tensor, 
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
        :param dones: Signals to indicate that episodes have ended
        :type dones: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, dones, timestep, timesteps)
        if self.memory is not None:
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)

        # track episodes internally
        if self._rollout:
            indexes = torch.nonzero(dones)
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
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = \
            self.memory.sample_all(names=self.tensors_names)[0]

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
            returns = torch.tensor(returns)
            return_threshold = torch.quantile(returns, self._percentile, dim=-1)
            
            # get elite states and actions
            indexes = torch.nonzero(returns >= return_threshold)
            elite_states = torch.cat([sampled_states[limits[i][0]:limits[i][1]] for i in indexes[:, 0]], dim=0)
            elite_actions = torch.cat([sampled_actions[limits[i][0]:limits[i][1]] for i in indexes[:, 0]], dim=0)

        # compute scores for the elite states
        scores = self.policy.act(elite_states)[2]

        # compute policy loss
        policy_loss = F.cross_entropy(scores, elite_actions.view(-1))

        # optimize policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # record data
        self.track_data("Loss / Policy loss", policy_loss.item())

        self.track_data("Coefficient / Return threshold", return_threshold.item())
        self.track_data("Coefficient / Mean discounted returns", torch.mean(returns).item())
