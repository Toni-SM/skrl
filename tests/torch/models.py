from typing import Sequence
from skrl.models.torch.base import Model
from skrl.models.torch import DeterministicMixin, SquashedGaussianMixin

import torch
from torch import nn as nn
from torchrl.modules import BatchRenorm1d


'''
Actor-Critic models for the CrossQ agent (with architectures almost identical to the ones used in the original paper)
'''

class Critic(DeterministicMixin, Model):
    net_arch: Sequence[int] = None
    use_batch_norm: bool = True

    batch_norm_momentum: float = 0.01
    batch_norm_epsilon: float = 0.001
    renorm_warmup_steps: int = 1e5

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        device=None,
        clip_actions=False,
        use_batch_norm=False,
        batch_norm_momentum=0.01,
        batch_norm_epsilon=0.001,
        renorm_warmup_steps: int = 1e5,
        **kwargs,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

        self.net_arch = net_arch
        self.use_batch_norm = use_batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_epsilon = batch_norm_epsilon
        self.renorm_warmup_steps = renorm_warmup_steps

        layers = []
        inputs = self.num_observations + self.num_actions
        if use_batch_norm:
            layers.append(BatchRenorm1d(inputs, momentum=batch_norm_momentum, eps=self.batch_norm_epsilon, warmup_steps=renorm_warmup_steps))
        layers.append(nn.Linear(inputs, net_arch[0]))
        layers.append(nn.ReLU())
        
        for i in range(len(net_arch) - 1):
            if use_batch_norm:
                layers.append(BatchRenorm1d(net_arch[i], momentum=batch_norm_momentum, eps=self.batch_norm_epsilon, warmup_steps=renorm_warmup_steps))
            layers.append(nn.Linear(net_arch[i], net_arch[i + 1]))
            layers.append(nn.ReLU())
            
        if use_batch_norm:
            layers.append(
                BatchRenorm1d(net_arch[-1], momentum=batch_norm_momentum, eps=self.batch_norm_epsilon, warmup_steps=renorm_warmup_steps)
            )
            
        layers.append(nn.Linear(net_arch[-1], 1))
        self.qnet = nn.Sequential(*layers)

    def compute(self, inputs, _):
        X = torch.cat((inputs["states"], inputs["taken_actions"]), dim=1)
        return self.qnet(X), {}
    
    def set_bn_training_mode(self, mode: bool) -> None:
        """
        Set the training mode of the BatchRenorm layers.
        When training is True, the running statistics are updated.

        :param mode: Whether to set the layers in training mode or not
        """
        for module in self.modules():
            if isinstance(module, BatchRenorm1d):
                module.train(mode)


class StochasticActor(SquashedGaussianMixin, Model):
    net_arch: Sequence[int] = None
    use_batch_norm: bool = True

    batch_norm_momentum: float = 0.01
    batch_norm_epsilon: float = 0.001
    renorm_warmup_steps: int = 1e5

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        use_batch_norm=False,
        batch_norm_momentum=0.01,
        batch_norm_epsilon=0.001,
        renorm_warmup_steps: int = 1e5,
        **kwargs,
    ):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        SquashedGaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        self.net_arch = net_arch
        self.use_batch_norm = use_batch_norm
        self.batch_norm_momentum = batch_norm_momentum
        self.batch_norm_epsilon = batch_norm_epsilon
        self.renorm_warmup_steps = renorm_warmup_steps

        layers = []
        inputs = self.num_observations
        if use_batch_norm:
            layers.append(BatchRenorm1d(inputs, momentum=batch_norm_momentum, eps=self.batch_norm_epsilon, warmup_steps=renorm_warmup_steps))
        layers.append(nn.Linear(inputs, net_arch[0]))
        layers.append(nn.ReLU())
        
        for i in range(len(net_arch) - 1):
            if use_batch_norm:
                layers.append(BatchRenorm1d(net_arch[i], momentum=batch_norm_momentum, eps=self.batch_norm_epsilon, warmup_steps=renorm_warmup_steps))
            layers.append(nn.Linear(net_arch[i], net_arch[i + 1]))
            layers.append(nn.ReLU())
                
        if use_batch_norm:
            layers.append(BatchRenorm1d(net_arch[-1], momentum=batch_norm_momentum, eps=self.batch_norm_epsilon, warmup_steps=renorm_warmup_steps))
            
        self.latent_pi = nn.Sequential(*layers)
        self.mu = nn.Linear(net_arch[-1], self.num_actions)
        self.log_std = nn.Linear(net_arch[-1], self.num_actions)

    def compute(self, inputs, _):
        latent_pi = self.latent_pi(inputs["states"])
        # print(f"obs: {inputs['states']}")
        # print(f"latent_pi: {latent_pi[0]}")
        return self.mu(latent_pi), self.log_std(latent_pi), {}
    
    def set_bn_training_mode(self, mode: bool) -> None:
        """
        Set the training mode of the BatchRenorm layers.
        When training is True, the running statistics are updated.

        :param mode: Whether to set the layers in training mode or not
        """
        for module in self.modules():
            if isinstance(module, BatchRenorm1d):
                module.train(mode)