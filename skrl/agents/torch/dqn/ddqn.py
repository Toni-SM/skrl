from typing import Union, Tuple, Dict, Any

import gym
import copy
import math

import torch
import torch.nn.functional as F

from ....memories.torch import Memory
from ....models.torch import Model

from .. import Agent


DDQN_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size
    
    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)
    
    "learning_rate": 1e-3,          # learning rate

    "random_timesteps": 1000,       # random exploration steps
    "learning_starts": 1000,        # learning starts after this many steps

    "update_interval": 1,           # agent update interval
    "target_update_interval": 10,   # target network update interval

    "exploration": {
        "initial_epsilon": 1.0,       # initial epsilon for epsilon-greedy exploration
        "final_epsilon": 0.05,        # final epsilon for epsilon-greedy exploration
        "timesteps": 1000,            # timesteps for epsilon-greedy decay 
    },

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "checkpoint_policy_only": True,     # checkpoint for policy only
    }
}


class DDQN(Agent):
    def __init__(self, 
                 models: Dict[str, Model], 
                 memory: Union[Memory, Tuple[Memory], None] = None, 
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 device: Union[str, torch.device] = "cuda:0", 
                 cfg: dict = {}) -> None:
        """Double Deep Q-Network (DDQN)

        https://ojs.aaai.org/index.php/AAAI/article/view/10295
        
        :param models: Models used by the agent
        :type models: dictionary of skrl.models.torch.Model
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

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(DDQN_DEFAULT_CONFIG)
        _cfg.update(cfg)
        super().__init__(models=models, 
                         memory=memory, 
                         observation_space=observation_space, 
                         action_space=action_space, 
                         device=device, 
                         cfg=_cfg)

        # models
        self.q_network = self.models.get("q_network", None)
        self.target_q_network = self.models.get("target_q_network", None)

        # checkpoint models
        self.checkpoint_models = {"q_network": self.q_network} if self.checkpoint_policy_only else self.models
        
        if self.target_q_network is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_q_network.freeze_parameters(True)

            # update target networks (hard update)
            self.target_q_network.update_parameters(self.q_network, polyak=1)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]
        
        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._learning_rate = self.cfg["learning_rate"]
        
        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._update_interval = self.cfg["update_interval"]
        self._target_update_interval = self.cfg["target_update_interval"]

        self._exploration_initial_epsilon = self.cfg["exploration"]["initial_epsilon"]
        self._exploration_final_epsilon = self.cfg["exploration"]["final_epsilon"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]
        
        # set up optimizers
        if self.q_network is not None:
            self.q_network_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self._learning_rate)

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
        :param inference: Flag to indicate whether the model is making inference
        :type inference: bool

        :return: Actions
        :rtype: torch.Tensor
        """
        if not self._exploration_timesteps:
            return torch.argmax(self.q_network.act(states, inference=inference)[0], dim=1, keepdim=True), None, None
            
        # sample random actions
        actions = self.q_network.random_act(states)[0]
        if timestep < self._random_timesteps:
            return actions, None, None

        # sample actions with epsilon-greedy policy
        epsilon = self._exploration_final_epsilon + (self._exploration_initial_epsilon - self._exploration_final_epsilon) \
                * math.exp(-1.0 * timestep / self._exploration_timesteps)

        indexes = (torch.rand(states.shape[0], device=self.device) >= epsilon).nonzero().view(-1)
        if indexes.numel():
            actions[indexes] = torch.argmax(self.q_network.act(states[indexes], inference=inference)[0], dim=1, keepdim=True)
        
        # record epsilon
        self.track_data("Exploration / Exploration epsilon", epsilon)
        
        return actions, None, None

    def record_transition(self, 
                          states: torch.Tensor, 
                          actions: torch.Tensor, 
                          rewards: torch.Tensor, 
                          next_states: torch.Tensor, 
                          dones: torch.Tensor, 
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
        :param dones: Signals to indicate that episodes have ended
        :type dones: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(states, actions, rewards, next_states, dones, infos, timestep, timesteps)
        if self.memory is not None:
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)

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
        if timestep >= self._learning_starts and not timestep % self._update_interval:
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
        # sample a batch from memory
        sampled_states, sampled_actions, sampled_rewards, sampled_next_states, sampled_dones = \
            self.memory.sample(names=self.tensors_names, batch_size=self._batch_size)[0]

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            # compute target values
            with torch.no_grad():
                next_q_values, _, _ = self.target_q_network.act(states=sampled_next_states)
                
                target_q_values = torch.gather(next_q_values, dim=1, \
                    index=torch.argmax(self.q_network.act(states=sampled_next_states)[0], dim=1, keepdim=True))
                target_values = sampled_rewards + self._discount_factor * sampled_dones.logical_not() * target_q_values

            # compute Q-network loss
            q_values = torch.gather(self.q_network.act(states=sampled_states)[0], dim=1, index=sampled_actions.long())

            q_network_loss = F.mse_loss(q_values, target_values)
            
            # optimize Q-network
            self.q_network_optimizer.zero_grad()
            q_network_loss.backward()
            self.q_network_optimizer.step()

            # update target network
            if not timestep % self._target_update_interval:
                self.target_q_network.update_parameters(self.q_network, polyak=self._polyak)

            # record data
            self.track_data("Loss / Q-network loss", q_network_loss.item())

            self.track_data("Target / Target (max)", torch.max(target_values).item())
            self.track_data("Target / Target (min)", torch.min(target_values).item())
            self.track_data("Target / Target (mean)", torch.mean(target_values).item())
