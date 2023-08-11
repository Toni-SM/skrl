from typing import Any, Callable, Dict, Optional, Tuple, Union

import copy
import itertools
import math
import gym
import gymnasium

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model


# [start-config-dict-torch]
AMP_DEFAULT_CONFIG = {
    "rollouts": 16,                 # number of rollouts before updating
    "learning_epochs": 6,           # number of learning epochs during each update
    "mini_batches": 2,              # number of mini batches during each learning epoch

    "discount_factor": 0.99,        # discount factor (gamma)
    "lambda": 0.95,                 # TD(lambda) coefficient (lam) for computing returns and advantages

    "learning_rate": 5e-5,                  # learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "value_preprocessor": None,             # value preprocessor class (see skrl.resources.preprocessors)
    "value_preprocessor_kwargs": {},        # value preprocessor's kwargs (e.g. {"size": 1})
    "amp_state_preprocessor": None,         # AMP state preprocessor class (see skrl.resources.preprocessors)
    "amp_state_preprocessor_kwargs": {},    # AMP state preprocessor's kwargs (e.g. {"size": env.amp_observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0.0,              # clipping coefficient for the norm of the gradients
    "ratio_clip": 0.2,                  # clipping coefficient for computing the clipped surrogate objective
    "value_clip": 0.2,                  # clipping coefficient for computing the value loss (if clip_predicted_values is True)
    "clip_predicted_values": False,     # clip predicted values during value loss computation

    "entropy_loss_scale": 0.0,          # entropy loss scaling factor
    "value_loss_scale": 2.5,            # value loss scaling factor
    "discriminator_loss_scale": 5.0,    # discriminator loss scaling factor

    "amp_batch_size": 512,                  # batch size for updating the reference motion dataset
    "task_reward_weight": 0.0,              # task-reward weight (wG)
    "style_reward_weight": 1.0,             # style-reward weight (wS)
    "discriminator_batch_size": 0,          # batch size for computing the discriminator loss (all samples if 0)
    "discriminator_reward_scale": 2,                    # discriminator reward scaling factor
    "discriminator_logit_regularization_scale": 0.05,   # logit regularization scale factor for the discriminator loss
    "discriminator_gradient_penalty_scale": 5,          # gradient penalty scaling factor for the discriminator loss
    "discriminator_weight_decay_scale": 0.0001,         # weight decay scaling factor for the discriminator loss

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward
    "time_limit_bootstrap": False,  # bootstrap at timeout termination (episode truncation)

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


class AMP(Agent):
    def __init__(self,
                 models: Dict[str, Model],
                 memory: Optional[Union[Memory, Tuple[Memory]]] = None,
                 observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 action_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 device: Optional[Union[str, torch.device]] = None,
                 cfg: Optional[dict] = None,
                 amp_observation_space: Optional[Union[int, Tuple[int], gym.Space, gymnasium.Space]] = None,
                 motion_dataset: Optional[Memory] = None,
                 reply_buffer: Optional[Memory] = None,
                 collect_reference_motions: Optional[Callable[[int], torch.Tensor]] = None,
                 collect_observation: Optional[Callable[[], torch.Tensor]] = None) -> None:
        """Adversarial Motion Priors (AMP)

        https://arxiv.org/abs/2104.02180

        The implementation is adapted from the NVIDIA IsaacGymEnvs
        (https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/learning/amp_continuous.py)

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
        :param amp_observation_space: AMP observation/state space or shape (default: ``None``)
        :type amp_observation_space: int, tuple or list of int, gym.Space, gymnasium.Space or None
        :param motion_dataset: Reference motion dataset: M (default: ``None``)
        :type motion_dataset: skrl.memory.torch.Memory or None
        :param reply_buffer: Reply buffer for preventing discriminator overfitting: B (default: ``None``)
        :type reply_buffer: skrl.memory.torch.Memory or None
        :param collect_reference_motions: Callable to collect reference motions (default: ``None``)
        :type collect_reference_motions: Callable[[int], torch.Tensor] or None
        :param collect_observation: Callable to collect observation (default: ``None``)
        :type collect_observation: Callable[[], torch.Tensor] or None

        :raises KeyError: If the models dictionary is missing a required key
        """
        _cfg = copy.deepcopy(AMP_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(models=models,
                         memory=memory,
                         observation_space=observation_space,
                         action_space=action_space,
                         device=device,
                         cfg=_cfg)

        self.amp_observation_space = amp_observation_space
        self.motion_dataset = motion_dataset
        self.reply_buffer = reply_buffer
        self.collect_reference_motions = collect_reference_motions
        self.collect_observation = collect_observation

        # models
        self.policy = self.models.get("policy", None)
        self.value = self.models.get("value", None)
        self.discriminator = self.models.get("discriminator", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["value"] = self.value
        self.checkpoint_modules["discriminator"] = self.discriminator

        # configuration
        self._learning_epochs = self.cfg["learning_epochs"]
        self._mini_batches = self.cfg["mini_batches"]
        self._rollouts = self.cfg["rollouts"]
        self._rollout = 0

        self._grad_norm_clip = self.cfg["grad_norm_clip"]
        self._ratio_clip = self.cfg["ratio_clip"]
        self._value_clip = self.cfg["value_clip"]
        self._clip_predicted_values = self.cfg["clip_predicted_values"]

        self._value_loss_scale = self.cfg["value_loss_scale"]
        self._entropy_loss_scale = self.cfg["entropy_loss_scale"]
        self._discriminator_loss_scale = self.cfg["discriminator_loss_scale"]

        self._learning_rate = self.cfg["learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]
        self._value_preprocessor = self.cfg["value_preprocessor"]
        self._amp_state_preprocessor = self.cfg["amp_state_preprocessor"]

        self._discount_factor = self.cfg["discount_factor"]
        self._lambda = self.cfg["lambda"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._amp_batch_size = self.cfg["amp_batch_size"]
        self._task_reward_weight = self.cfg["task_reward_weight"]
        self._style_reward_weight = self.cfg["style_reward_weight"]

        self._discriminator_batch_size = self.cfg["discriminator_batch_size"]
        self._discriminator_reward_scale = self.cfg["discriminator_reward_scale"]
        self._discriminator_logit_regularization_scale = self.cfg["discriminator_logit_regularization_scale"]
        self._discriminator_gradient_penalty_scale = self.cfg["discriminator_gradient_penalty_scale"]
        self._discriminator_weight_decay_scale = self.cfg["discriminator_weight_decay_scale"]

        self._rewards_shaper = self.cfg["rewards_shaper"]
        self._time_limit_bootstrap = self.cfg["time_limit_bootstrap"]

        # set up optimizer and learning rate scheduler
        if self.policy is not None and self.value is not None and self.discriminator is not None:
            self.optimizer = torch.optim.Adam(itertools.chain(self.policy.parameters(),
                                                              self.value.parameters(),
                                                              self.discriminator.parameters()),
                                              lr=self._learning_rate)
            if self._learning_rate_scheduler is not None:
                self.scheduler = self._learning_rate_scheduler(self.optimizer, **self.cfg["learning_rate_scheduler_kwargs"])

            self.checkpoint_modules["optimizer"] = self.optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(**self.cfg["state_preprocessor_kwargs"])
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

        if self._value_preprocessor:
            self._value_preprocessor = self._value_preprocessor(**self.cfg["value_preprocessor_kwargs"])
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

        if self._amp_state_preprocessor:
            self._amp_state_preprocessor = self._amp_state_preprocessor(**self.cfg["amp_state_preprocessor_kwargs"])
            self.checkpoint_modules["amp_state_preprocessor"] = self._amp_state_preprocessor
        else:
            self._amp_state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the agent
        """
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(name="states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_states", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)

            self.memory.create_tensor(name="amp_states", size=self.amp_observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="next_values", size=1, dtype=torch.float32)

        self.tensors_names = ["states", "actions", "rewards", "next_states", "terminated", \
            "log_prob", "values", "returns", "advantages", "amp_states", "next_values"]

        # create tensors for motion dataset and reply buffer
        if self.motion_dataset is not None:
            self.motion_dataset.create_tensor(name="states", size=self.amp_observation_space, dtype=torch.float32)
            self.reply_buffer.create_tensor(name="states", size=self.amp_observation_space, dtype=torch.float32)

            # initialize motion dataset
            for _ in range(math.ceil(self.motion_dataset.memory_size / self._amp_batch_size)):
                self.motion_dataset.add_samples(states=self.collect_reference_motions(self._amp_batch_size))

        # create temporary variables needed for storage and computation
        self._current_log_prob = None
        self._current_states = None

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
        # use collected states
        if self._current_states is not None:
            states = self._current_states

        states = self._state_preprocessor(states)

        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": states}, role="policy")

        # sample stochastic actions
        actions, log_prob, outputs = self.policy.act({"states": states}, role="policy")
        self._current_log_prob = log_prob

        return actions, log_prob, outputs

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
        # use collected states
        if self._current_states is not None:
            states = self._current_states

        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)

        if self.memory is not None:
            amp_states = infos["amp_obs"]

            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            with torch.no_grad():
                values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
            values = self._value_preprocessor(values, inverse=True)

            # time-limit (truncation) boostrapping
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated

            with torch.no_grad():
                next_values, _, _ = self.value.act({"states": self._state_preprocessor(next_states)}, role="value")
            next_values = self._value_preprocessor(next_values, inverse=True)
            next_values *= infos['terminate'].view(-1, 1).logical_not()

            self.memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, terminated=terminated, truncated=truncated,
                                    log_prob=self._current_log_prob, values=values, amp_states=amp_states, next_values=next_values)
            for memory in self.secondary_memories:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states, terminated=terminated, truncated=truncated,
                                   log_prob=self._current_log_prob, values=values, amp_states=amp_states, next_values=next_values)

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if self.collect_observation is not None:
            self._current_states = self.collect_observation()

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        self._rollout += 1
        if not self._rollout % self._rollouts and timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        def compute_gae(rewards: torch.Tensor,
                        dones: torch.Tensor,
                        values: torch.Tensor,
                        next_values: torch.Tensor,
                        discount_factor: float = 0.99,
                        lambda_coefficient: float = 0.95) -> torch.Tensor:
            """Compute the Generalized Advantage Estimator (GAE)

            :param rewards: Rewards obtained by the agent
            :type rewards: torch.Tensor
            :param dones: Signals to indicate that episodes have ended
            :type dones: torch.Tensor
            :param values: Values obtained by the agent
            :type values: torch.Tensor
            :param next_values: Next values obtained by the agent
            :type next_values: torch.Tensor
            :param discount_factor: Discount factor
            :type discount_factor: float
            :param lambda_coefficient: Lambda coefficient
            :type lambda_coefficient: float

            :return: Generalized Advantage Estimator
            :rtype: torch.Tensor
            """
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            memory_size = rewards.shape[0]

            # advantages computation
            for i in reversed(range(memory_size)):
                advantage = rewards[i] - values[i] + discount_factor * (next_values[i] + lambda_coefficient * not_dones[i] * advantage)
                advantages[i] = advantage
            # returns computation
            returns = advantages + values
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            return returns, advantages

        # update dataset of reference motions
        self.motion_dataset.add_samples(states=self.collect_reference_motions(self._amp_batch_size))

        # compute combined rewards
        rewards = self.memory.get_tensor_by_name("rewards")
        amp_states = self.memory.get_tensor_by_name("amp_states")

        with torch.no_grad():
            amp_logits, _, _ = self.discriminator.act({"states": self._amp_state_preprocessor(amp_states)}, role="discriminator")
            style_reward = -torch.log(torch.maximum(1 - 1 / (1 + torch.exp(-amp_logits)), torch.tensor(0.0001, device=self.device)))
            style_reward *= self._discriminator_reward_scale

        combined_rewards = self._task_reward_weight * rewards + self._style_reward_weight * style_reward

        # compute returns and advantages
        values = self.memory.get_tensor_by_name("values")
        next_values=self.memory.get_tensor_by_name("next_values")
        returns, advantages = compute_gae(rewards=combined_rewards,
                                          dones=self.memory.get_tensor_by_name("terminated"),
                                          values=values,
                                          next_values=next_values,
                                          discount_factor=self._discount_factor,
                                          lambda_coefficient=self._lambda)

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # sample mini-batches from memory
        sampled_batches = self.memory.sample_all(names=self.tensors_names, mini_batches=self._mini_batches)
        sampled_motion_batches = self.motion_dataset.sample(names=["states"],
                                                            batch_size=self.memory.memory_size * self.memory.num_envs,
                                                            mini_batches=self._mini_batches)
        if len(self.reply_buffer):
            sampled_replay_batches = self.reply_buffer.sample(names=["states"],
                                                              batch_size=self.memory.memory_size * self.memory.num_envs,
                                                              mini_batches=self._mini_batches)
        else:
            sampled_replay_batches = [[batches[self.tensors_names.index("amp_states")]] for batches in sampled_batches]

        cumulative_policy_loss = 0
        cumulative_entropy_loss = 0
        cumulative_value_loss = 0
        cumulative_discriminator_loss = 0

        # learning epochs
        for epoch in range(self._learning_epochs):

            # mini-batches loop
            for batch_index, (sampled_states, sampled_actions, _, _, _, \
                sampled_log_prob, sampled_values, sampled_returns, sampled_advantages, \
                sampled_amp_states, _) in enumerate(sampled_batches):

                sampled_states = self._state_preprocessor(sampled_states, train=True)

                _, next_log_prob, _ = self.policy.act({"states": sampled_states, "taken_actions": sampled_actions}, role="policy")

                # compute entropy loss
                if self._entropy_loss_scale:
                    entropy_loss = -self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                else:
                    entropy_loss = 0

                # compute policy loss
                ratio = torch.exp(next_log_prob - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)

                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                # compute value loss
                predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")

                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                min=-self._value_clip,
                                                                max=self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                # compute discriminator loss
                if self._discriminator_batch_size:
                    sampled_amp_states = self._amp_state_preprocessor(sampled_amp_states[0:self._discriminator_batch_size], train=True)
                    sampled_amp_replay_states = self._amp_state_preprocessor(
                        sampled_replay_batches[batch_index][0][0:self._discriminator_batch_size], train=True)
                    sampled_amp_motion_states = self._amp_state_preprocessor(
                        sampled_motion_batches[batch_index][0][0:self._discriminator_batch_size], train=True)
                else:
                    sampled_amp_states = self._amp_state_preprocessor(sampled_amp_states, train=True)
                    sampled_amp_replay_states = self._amp_state_preprocessor(sampled_replay_batches[batch_index][0], train=True)
                    sampled_amp_motion_states = self._amp_state_preprocessor(sampled_motion_batches[batch_index][0], train=True)

                sampled_amp_motion_states.requires_grad_(True)
                amp_logits, _, _ = self.discriminator.act({"states": sampled_amp_states}, role="discriminator")
                amp_replay_logits, _, _ = self.discriminator.act({"states": sampled_amp_replay_states}, role="discriminator")
                amp_motion_logits, _, _ = self.discriminator.act({"states": sampled_amp_motion_states}, role="discriminator")

                amp_cat_logits = torch.cat([amp_logits, amp_replay_logits], dim=0)

                # discriminator prediction loss
                discriminator_loss = 0.5 * (nn.BCEWithLogitsLoss()(amp_cat_logits, torch.zeros_like(amp_cat_logits)) \
                    + torch.nn.BCEWithLogitsLoss()(amp_motion_logits, torch.ones_like(amp_motion_logits)))

                # discriminator logit regularization
                if self._discriminator_logit_regularization_scale:
                    logit_weights = torch.flatten(list(self.discriminator.modules())[-1].weight)
                    discriminator_loss += self._discriminator_logit_regularization_scale * torch.sum(torch.square(logit_weights))

                # discriminator gradient penalty
                if self._discriminator_gradient_penalty_scale:
                    amp_motion_gradient = torch.autograd.grad(amp_motion_logits,
                                                              sampled_amp_motion_states,
                                                              grad_outputs=torch.ones_like(amp_motion_logits),
                                                              create_graph=True,
                                                              retain_graph=True,
                                                              only_inputs=True)
                    gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
                    discriminator_loss += self._discriminator_gradient_penalty_scale * gradient_penalty

                # discriminator weight decay
                if self._discriminator_weight_decay_scale:
                    weights = [torch.flatten(module.weight) for module in self.discriminator.modules() \
                        if isinstance(module, torch.nn.Linear)]
                    weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
                    discriminator_loss += self._discriminator_weight_decay_scale * weight_decay

                discriminator_loss *= self._discriminator_loss_scale

                # optimization step
                self.optimizer.zero_grad()
                (policy_loss + entropy_loss + value_loss + discriminator_loss).backward()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(itertools.chain(self.policy.parameters(),
                                                             self.value.parameters(),
                                                             self.discriminator.parameters()), self._grad_norm_clip)
                self.optimizer.step()

                # update cumulative losses
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
                cumulative_discriminator_loss += discriminator_loss.item()

            # update learning rate
            if self._learning_rate_scheduler:
                self.scheduler.step()

        # update AMP repaly buffer
        self.reply_buffer.add_samples(states=amp_states.view(-1, amp_states.shape[-1]))

        # record data
        self.track_data("Loss / Policy loss", cumulative_policy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Value loss", cumulative_value_loss / (self._learning_epochs * self._mini_batches))
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / (self._learning_epochs * self._mini_batches))
        self.track_data("Loss / Discriminator loss", cumulative_discriminator_loss / (self._learning_epochs * self._mini_batches))

        self.track_data("Policy / Standard deviation", self.policy.distribution(role="policy").stddev.mean().item())

        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
