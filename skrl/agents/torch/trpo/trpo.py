from typing import Union, Tuple, Dict

import gym
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector
from torch.nn.utils.convert_parameters import vector_to_parameters

from ....memories.torch import Memory
from ....models.torch import Model

from .. import Agent


TRPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch
    
    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.99,                 # TD(lambda) coefficient (lam) for computing returns and advantages
    
    "value_learning_rate": 1e-3,    # value learning rate

    "random_timesteps": 1000,       # random exploration steps
    "learning_starts": 1000,        # learning starts after this many steps

    "grad_norm_clip": 0.5,          # clipping coefficient for the norm of the gradients
    "value_loss_scale": 1.0,        # value loss scaling factor

    "damping": 0.1,                     # damping coefficient for computing the Hessian-vector product
    "max_kl_divergence": 0.01,          # maximum KL divergence between old and new policy
    "conjugate_gradient_steps": 10,     # maximum number of iterations for the conjugate gradient algorithm
    "max_backtrack_steps": 10,          # maximum number of backtracking steps during line search
    "accept_ratio": 0.5,                # accept ratio for the line search loss improvement
    "step_fraction": 1.0,               # fraction of the step size for the line search

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": 250,      # TensorBoard writing interval (timesteps)

        "checkpoint_interval": 1000,        # interval for checkpoints (timesteps)
        "checkpoint_policy_only": True,     # checkpoint for policy only
    }
}


class TRPO(Agent):
    def __init__(self, 
                 models: Dict[str, Model], 
                 memory: Union[Memory, Tuple[Memory], None] = None, 
                 observation_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 action_space: Union[int, Tuple[int], gym.Space, None] = None, 
                 device: Union[str, torch.device] = "cuda:0", 
                 cfg: dict = {}) -> None:
        """Trust Region Policy Optimization (TRPO)

        https://arxiv.org/abs/1502.05477
        
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
        _cfg = copy.deepcopy(TRPO_DEFAULT_CONFIG)
        _cfg.update(cfg)
        super().__init__(models=models, 
                         memory=memory, 
                         observation_space=observation_space, 
                         action_space=action_space, 
                         device=device, 
                         cfg=_cfg)

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        self.backup_policy = copy.deepcopy(self.policy)

        # checkpoint models
        self.checkpoint_models = {"policy": self.policy} if self.checkpoint_policy_only else self.models

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._value_loss_scale = self.cfg["value_loss_scale"]

        self._max_kl_divergence = self.cfg["max_kl_divergence"]
        self._damping = self.cfg["damping"]
        self._conjugate_gradient_steps = self.cfg["conjugate_gradient_steps"]
        self._max_backtrack_steps = self.cfg["max_backtrack_steps"]
        self._accept_ratio = self.cfg["accept_ratio"]
        self._step_fraction = self.cfg["step_fraction"]

        self._value_learning_rate = self.cfg["value_learning_rate"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        # set up optimizers
        if self.policy is not None and self.value is not None:
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self._value_learning_rate)

    def init(self) -> None:
        """Initialize the agent
        """
        super().init()
        
        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="dones", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

        self.tensors_names = ["states", "actions", "log_prob", "returns", "advantages"]

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_next_states = None

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
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act(states)

        # sample stochastic actions
        actions, log_prob, actions_mean = self.policy.act(states, inference=inference)
        self._current_log_prob = log_prob

        return actions, log_prob, actions_mean

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

        self._current_next_states = next_states

        if self.memory is not None:
            values, _, _ = self.value.act(states=states, inference=True)
            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones, 
                                    log_prob=self._current_log_prob, values=values)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones, 
                                   log_prob=self._current_log_prob, values=values)

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

        def surrogate_loss(policy: Model, 
                           states: torch.Tensor, 
                           actions: torch.Tensor, 
                           log_prob: torch.Tensor, 
                           advantages: torch.Tensor) -> torch.Tensor:
            """Compute the surrogate objective (policy loss)

            :param policy: Policy
            :type policy: Model
            :param states: States
            :type states: torch.Tensor
            :param actions: Actions
            :type actions: torch.Tensor
            :param log_prob: Log probability
            :type log_prob: torch.Tensor
            :param advantages: Advantages
            :type advantages: torch.Tensor

            :return: Surrogate loss
            :rtype: torch.Tensor
            """
            _, new_log_prob, _ = policy.act(states, actions)
            return (advantages * torch.exp(new_log_prob - log_prob.detach())).mean()

        def conjugate_gradient(policy: Model, 
                               states: torch.Tensor, 
                               b: torch.Tensor, 
                               num_iterations: float = 10, 
                               residual_tolerance: float = 1e-10) -> torch.Tensor:
            """Conjugate gradient algorithm to solve Ax = b using the iterative method

            https://en.wikipedia.org/wiki/Conjugate_gradient_method#As_an_iterative_method

            :param policy: Policy
            :type policy: Model
            :param states: States
            :type states: torch.Tensor
            :param b: Vector b 
            :type b: torch.Tensor
            :param num_iterations: Number of iterations (default: 10)
            :type num_iterations: float, optional
            :param residual_tolerance: Residual tolerance (default: 1e-10)
            :type residual_tolerance: float, optional

            :return: Conjugate vector
            :rtype: torch.Tensor
            """
            x = torch.zeros_like(b)
            r = b.clone()
            p = b.clone()
            rr_old = torch.dot(r, r)
            for _ in range(num_iterations):
                hv = fisher_vector_product(policy, states, p, damping=self._damping)
                alpha = rr_old / torch.dot(p, hv)
                x += alpha * p
                r -= alpha * hv
                rr_new = torch.dot(r, r)
                if rr_new < residual_tolerance:
                    break
                p = r + rr_new / rr_old * p
                rr_old = rr_new
            return x

        def fisher_vector_product(policy: Model, 
                                  states: torch.Tensor, 
                                  vector: torch.Tensor, 
                                  damping: float = 0.1) -> torch.Tensor:
            """Compute the Fisher vector product (direct method)
            
            https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/

            :param policy: Policy
            :type policy: Model
            :param states: States
            :type states: torch.Tensor
            :param vector: Vector
            :type vector: torch.Tensor
            :param damping: Damping (default: 0.1)
            :type damping: float, optional

            :return: Hessian vector product
            :rtype: torch.Tensor
            """
            kl = kl_divergence(policy, policy, states)
            kl_gradient = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
            flat_kl_gradient = torch.cat([gradient.view(-1) for gradient in kl_gradient])
            hessian_vector_gradient = torch.autograd.grad((flat_kl_gradient * vector).sum(), policy.parameters())
            flat_hessian_vector_gradient = torch.cat([gradient.contiguous().view(-1) for gradient in hessian_vector_gradient])
            return flat_hessian_vector_gradient + damping * vector

        def kl_divergence(policy_1: Model, policy_2: Model, states: torch.Tensor) -> torch.Tensor:
            """Compute the KL divergence between two distributions

            https://en.wikipedia.org/wiki/Normal_distribution#Other_properties

            :param policy_1: First policy
            :type policy_1: Model
            :param policy_2: Second policy
            :type policy_2: Model
            :param states: States
            :type states: torch.Tensor

            :return: KL divergence
            :rtype: torch.Tensor
            """
            _, _, mu_1 = policy_1.act(states)
            logstd_1 = policy_1.get_log_std()
            mu_1, logstd_1 = mu_1.detach(), logstd_1.detach()

            _, _, mu_2 = policy_2.act(states)
            logstd_2 = policy_2.get_log_std()
            
            kl = logstd_1 - logstd_2 + 0.5 * (torch.square(logstd_1.exp()) + torch.square(mu_1 - mu_2)) \
               / torch.square(logstd_2.exp()) - 0.5
            return torch.sum(kl, dim=-1).mean()

        # compute returns and advantages
        last_values, _, _ = self.value.act(states=self._current_next_states.float() \
            if not torch.is_floating_point(self._current_next_states) else self._current_next_states, inference=True)
        
        computing_hyperparameters = {"discount_factor": self._discount_factor,
                                     "lambda_coefficient": self._lambda,
                                     "normalize_returns": False,
                                     "normalize_advantages": True}
        self.memory.compute_functions(returns_dst="returns", 
                                      advantages_dst="advantages", 
                                      last_values=last_values, 
                                      hyperparameters=computing_hyperparameters)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self.tensors_names, mini_batches=self._mini_batches)

        cumulative_policy_loss = 0
        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):
            
            # mini-batches loop
            for sampled_states, sampled_actions, sampled_log_prob, sampled_returns, sampled_advantages in sampled_batches:

                # compute policy loss gradient
                policy_loss = surrogate_loss(self.policy, sampled_states, sampled_actions, sampled_log_prob, sampled_advantages)
                policy_loss_gradient = torch.autograd.grad(policy_loss, self.policy.parameters())
                flat_policy_loss_gradient = torch.cat([gradient.view(-1) for gradient in policy_loss_gradient])

                # compute the search direction using the conjugate gradient algorithm
                search_direction = conjugate_gradient(self.policy, sampled_states, flat_policy_loss_gradient.data, 
                                                      num_iterations=self._conjugate_gradient_steps)

                # compute step size and full step
                xHx = (search_direction * fisher_vector_product(self.policy, sampled_states, search_direction, self._damping)) \
                    .sum(0, keepdim=True)
                step_size = torch.sqrt(2 * self._max_kl_divergence / xHx)[0]
                full_step = step_size * search_direction

                # backtracking line search
                restore_policy_flag = True
                self.backup_policy.update_parameters(self.policy)
                params = parameters_to_vector(self.policy.parameters())

                expected_improvement = (flat_policy_loss_gradient * full_step).sum(0, keepdim=True)

                for alpha in [self._step_fraction * 0.5 ** i for i in range(self._max_backtrack_steps)]:
                    new_params = params + alpha * full_step
                    vector_to_parameters(new_params, self.policy.parameters())

                    expected_improvement *= alpha
                    kl = kl_divergence(self.backup_policy, self.policy, sampled_states)
                    loss = surrogate_loss(self.policy, sampled_states, sampled_actions, sampled_log_prob, sampled_advantages)

                    if kl < self._max_kl_divergence and (loss - policy_loss) / expected_improvement > self._accept_ratio:
                        restore_policy_flag = False
                        break

                if restore_policy_flag:
                    self.policy.update_parameters(self.backup_policy)

                # compute value loss
                predicted_values, _, _ = self.value.act(sampled_states)

                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # optimize value
                self.value_optimizer.zero_grad()
                value_loss.backward()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.value.parameters(), self._grad_norm_clip)
                self.value_optimizer.step()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / self._learning_epochs)
        self.track_data("Loss / Value loss", cumulative_value_loss / self._learning_epochs)
        
        self.track_data("Policy / Standard deviation", self.policy.distribution().stddev.mean().item())
