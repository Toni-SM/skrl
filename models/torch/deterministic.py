import torch

from . import Model


class DeterministicModel(Model):
    def __init__(self, env, device) -> None:
        """
        Deterministic model (Deterministic)

        # TODO: describe internal properties
        """
        super().__init__(env, device)
        
    def act(self, states, taken_actions=None, inference=False):
        # map from states/observations to actions
        actions = self.compute(states, taken_actions)

        return actions, torch.Tensor(), torch.Tensor()
        