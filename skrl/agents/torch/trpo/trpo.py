from typing import Any, Mapping, Optional, Tuple, Union

import copy
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.utils import ScopedTimer


# fmt: off
# [start-config-dict-torch]
TRPO_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 8,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "value_learning_rate": 1e-3,            # value learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "observation_preprocessor": None,       # observation preprocessor class (see skrl.resources.preprocessors)
    "observation_preprocessor_kwargs": {},  # observation preprocessor's kwargs (e.g. {"size": env.observation_space})
    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.state_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.5,          # clipping coefficient for the norm of the gradients
    "value_loss_scale": 1.0,        # value loss scaling factor

    "damping": 0.1,                     # damping coefficient for computing the Hessian-vector product
    "max_kl_divergence": 0.01,          # maximum KL divergence between old and new policy
    "conjugate_gradient_steps": 10,     # maximum number of iterations for the conjugate gradient algorithm
    "max_backtrack_steps": 10,          # maximum number of backtracking steps during line search
    "accept_ratio": 0.5,                # accept ratio for the line search loss improvement
    "step_fraction": 1.0,               # fraction of the step size for the line search

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


def compute_gae(
    *,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> torch.Tensor:
    """Compute the Generalized Advantage Estimator (GAE).

    :param rewards: Rewards obtained by the agent.
    :param dones: Signals to indicate that episodes have ended.
    :param values: Values obtained by the agent.
    :param next_values: Next values obtained by the agent.
    :param discount_factor: Discount factor.
    :param lambda_coefficient: Lambda coefficient.

    :return: Generalized Advantage Estimator.
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_dones = dones.logical_not()
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else next_values
        advantage = (
            rewards[i] - values[i] + discount_factor * not_dones[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages


def surrogate_loss(
    *,
    policy: Model,
    observations: torch.Tensor,
    states: torch.Tensor,
    actions: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
) -> torch.Tensor:
    """Compute the surrogate objective (policy loss).

    :param policy: Policy.
    :param observations: Observations.
    :param states: States.
    :param actions: Actions.
    :param log_prob: Log probability.
    :param advantages: Advantages.

    :return: Surrogate loss.
    """
    _, outputs = policy.act({"observations": observations, "states": states, "taken_actions": actions}, role="policy")
    new_log_prob = outputs["log_prob"]
    return (advantages * torch.exp(new_log_prob - log_prob.detach())).mean()


def conjugate_gradient(
    *,
    policy: Model,
    observations: torch.Tensor,
    states: torch.Tensor,
    b: torch.Tensor,
    damping: float = 0.1,
    num_iterations: float = 10,
    residual_tolerance: float = 1e-10,
) -> torch.Tensor:
    """Conjugate gradient algorithm to solve Ax = b using the iterative method.

    https://en.wikipedia.org/wiki/Conjugate_gradient_method#As_an_iterative_method

    :param policy: Policy.
    :param observations: Observations.
    :param states: States.
    :param b: Vector b.
    :param damping: Damping.
    :param num_iterations: Number of iterations.
    :param residual_tolerance: Residual tolerance.

    :return: Conjugate vector.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rr_old = torch.dot(r, r)
    for _ in range(num_iterations):
        hv = fisher_vector_product(policy=policy, observations=observations, states=states, vector=p, damping=damping)
        alpha = rr_old / torch.dot(p, hv)
        x += alpha * p
        r -= alpha * hv
        rr_new = torch.dot(r, r)
        if rr_new < residual_tolerance:
            break
        p = r + rr_new / rr_old * p
        rr_old = rr_new
    return x


def fisher_vector_product(
    *, policy: Model, observations: torch.Tensor, states: torch.Tensor, vector: torch.Tensor, damping: float = 0.1
) -> torch.Tensor:
    """Compute the Fisher vector product (direct method).

    https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/

    :param policy: Policy.
    :param observations: Observations.
    :param states: States.
    :param vector: Vector.
    :param damping: Damping.

    :return: Hessian vector product.
    """
    kl = kl_divergence(policy_1=policy, policy_2=policy, observations=observations, states=states)
    kl_gradient = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
    flat_kl_gradient = torch.cat([gradient.view(-1) for gradient in kl_gradient])
    hessian_vector_gradient = torch.autograd.grad((flat_kl_gradient * vector).sum(), policy.parameters())
    flat_hessian_vector_gradient = torch.cat([gradient.contiguous().view(-1) for gradient in hessian_vector_gradient])
    return flat_hessian_vector_gradient + damping * vector


def kl_divergence(
    *, policy_1: Model, policy_2: Model, observations: torch.Tensor, states: torch.Tensor
) -> torch.Tensor:
    """Compute the KL divergence between two distributions.

    https://en.wikipedia.org/wiki/Normal_distribution#Other_properties

    :param policy_1: First policy.
    :param policy_2: Second policy.
    :param observations: Observations.
    :param states: States.

    :return: KL divergence.
    """
    _, outputs = policy_1.act({"observations": observations, "states": states}, role="policy")
    mu_1 = outputs["mean_actions"].detach()
    logstd_1 = outputs["log_std"].detach()

    _, outputs = policy_2.act({"observations": observations, "states": states}, role="policy")
    mu_2 = outputs["mean_actions"]
    logstd_2 = outputs["log_std"]

    kl = (
        logstd_1
        - logstd_2
        + 0.5 * (torch.square(logstd_1.exp()) + torch.square(mu_1 - mu_2)) / torch.square(logstd_2.exp())
        - 0.5
    )
    return torch.sum(kl, dim=-1).mean()


class TRPO(Agent):
    def __init__(
        self,
        *,
        models: Optional[Mapping[str, Model]] = None,
        memory: Optional[Memory] = None,
        observation_space: Optional[gymnasium.Space] = None,
        state_space: Optional[gymnasium.Space] = None,
        action_space: Optional[gymnasium.Space] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        """Trust Region Policy Optimization (TRPO).

        https://arxiv.org/abs/1502.05477

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        _cfg = copy.deepcopy(TRPO_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)

        self.backup_policy = copy.deepcopy(self.policy)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.value is not None:
                self.value.broadcast_parameters()

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
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._observation_preprocessor = self.cfg["observation_preprocessor"]
        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None:
            self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self._value_learning_rate)
            if self._learning_rate_scheduler is not None:
                self.value_scheduler = self._learning_rate_scheduler(
                    self.value_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["value_optimizer"] = self.value_optimizer

        # set up preprocessors
        # - observations
        if self._observation_preprocessor:
            self._observation_preprocessor = self._observation_preprocessor(
                **self.cfg["observation_preprocessor_kwargs"]
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor
        # - states
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
        # - values
        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

    def init(self, *, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent.

        :param trainer_cfg: Trainer configuration.
        """
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            self._tensors_names_policy = ["observations", "states", "actions", "log_prob", "advantages"]
            self._tensors_names_value = ["observations", "states", "returns"]

        # create temporary variables needed for storage and computation
        self._current_next_observations = None
        self._current_next_states = None
        self._current_log_prob = None

    def act(
        self, observations: torch.Tensor, states: Union[torch.Tensor, None], *, timestep: int, timesteps: int
    ) -> Tuple[torch.Tensor, Mapping[str, Union[torch.Tensor, Any]]]:
        """Process the environment's observations/states to make a decision (actions) using the main policy.

        :param observations: Environment observations.
        :param states: Environment states.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.

        :return: Agent output. The first component is the expected action/value returned by the agent.
            The second component is a dictionary containing extra output values according to the model.
        """
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }
        # sample random actions
        # TODO: check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act(inputs, role="policy")

        # sample stochastic actions
        actions, outputs = self.policy.act(inputs, role="policy")
        self._current_log_prob = outputs["log_prob"]

        return actions, outputs

    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory.

        :param observations: Environment observations.
        :param states: Environment states.
        :param actions: Actions taken by the agent.
        :param rewards: Instant rewards achieved by the current actions.
        :param next_observations: Next environment observations.
        :param next_states: Next environment states.
        :param terminated: Signals that indicate episodes have terminated.
        :param truncated: Signals that indicate episodes have been truncated.
        :param infos: Additional information about the environment.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if self.memory is not None:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # compute values
            inputs = {
                "observations": self._observation_preprocessor(observations),
                "states": self._state_preprocessor(states),
            }
            values, _ = self.value.act(inputs, role="value")
            values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) bootstrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            # storage transition in memory
            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=values,
            )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called after the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            with ScopedTimer() as timer:
                self.enable_models_training_mode(True)
                self.update(timestep=timestep, timesteps=timesteps)
                self.enable_models_training_mode(False)
                self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)

        # write tracking data and checkpoints
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        # compute returns and advantages
        with torch.no_grad():
            inputs = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            dones=self.memory.get_tensor_by_name("terminated") | self.memory.get_tensor_by_name("truncated"),
            values=values,
            next_values=last_values,
            discount_factor=self._discount_factor,
            lambda_coefficient=self._lambda,
        )

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample all from memory
        sampled_observations, sampled_states, sampled_actions, sampled_log_prob, sampled_advantages = (
            self.memory.sample_all(names=self._tensors_names_policy, mini_batches=1)[0]
        )

        sampled_observations = self._observation_preprocessor(sampled_observations, train=True)
        sampled_states = self._state_preprocessor(sampled_states, train=True)

        # compute policy loss gradient
        policy_loss = surrogate_loss(
            policy=self.policy,
            observations=sampled_observations,
            states=sampled_states,
            actions=sampled_actions,
            log_prob=sampled_log_prob,
            advantages=sampled_advantages,
        )
        policy_loss_gradient = torch.autograd.grad(policy_loss, self.policy.parameters())
        flat_policy_loss_gradient = torch.cat([gradient.view(-1) for gradient in policy_loss_gradient])

        # compute the search direction using the conjugate gradient algorithm
        search_direction = conjugate_gradient(
            policy=self.policy,
            observations=sampled_observations,
            states=sampled_states,
            b=flat_policy_loss_gradient.data,
            damping=self._damping,
            num_iterations=self._conjugate_gradient_steps,
        )

        # compute step size and full step
        xHx = (
            search_direction
            * fisher_vector_product(
                policy=self.policy,
                observations=sampled_observations,
                states=sampled_states,
                vector=search_direction,
                damping=self._damping,
            )
        ).sum(0, keepdim=True)
        step_size = torch.sqrt(2 * self._max_kl_divergence / xHx)[0]
        full_step = step_size * search_direction

        # backtracking line search
        restore_policy_flag = True
        self.backup_policy.update_parameters(self.policy)
        params = parameters_to_vector(self.policy.parameters())

        expected_improvement = (flat_policy_loss_gradient * full_step).sum(0, keepdim=True)

        for alpha in [self._step_fraction * 0.5**i for i in range(self._max_backtrack_steps)]:
            new_params = params + alpha * full_step
            vector_to_parameters(new_params, self.policy.parameters())

            expected_improvement *= alpha
            kl = kl_divergence(
                policy_1=self.backup_policy,
                policy_2=self.policy,
                observations=sampled_observations,
                states=sampled_states,
            )
            loss = surrogate_loss(
                policy=self.policy,
                observations=sampled_observations,
                states=sampled_states,
                actions=sampled_actions,
                log_prob=sampled_log_prob,
                advantages=sampled_advantages,
            )

            if kl < self._max_kl_divergence and (loss - policy_loss) / expected_improvement > self._accept_ratio:
                restore_policy_flag = False
                break

        if restore_policy_flag:
            self.policy.update_parameters(self.backup_policy)

        if config.torch.is_distributed:
            self.policy.reduce_parameters()

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self._tensors_names_value, mini_batches=self._mini_batches)

        cumulative_value_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):

            # mini-batches loop
            for sampled_observations, sampled_states, sampled_returns in sampled_batches:

                inputs = {
                    "observations": self._observation_preprocessor(sampled_observations, train=not epoch),
                    "states": self._state_preprocessor(sampled_states, train=not epoch),
                }

                # compute value loss
                predicted_values, _ = self.value.act(inputs, role="value")

                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # optimization step (value)
                self.value_optimizer.zero_grad()
                value_loss.backward()
                if config.torch.is_distributed:
                    self.value.reduce_parameters()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self.value.parameters(), self._grad_norm_clip)
                self.value_optimizer.step()

                # update cumulative losses
                cumulative_value_loss += value_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                self.value_scheduler.step()

        # record data
        self.track_data("Loss / Policy loss", policy_loss.item())
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Value learning rate", self.value_scheduler.get_last_lr()[0])
