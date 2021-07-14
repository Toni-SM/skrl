import gym
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.network = None
        
    def forward(self):
        raise NotImplementedError 

    def act(self, state, inference=False):
        raise NotImplementedError

    def save(self, path):
        torch.save(self.network.state_dict(), path)

    def load(self, path):
        self.network.load_state_dict(torch.load(path))
    
    def activation_funtion_by_name(self, name):
        # TODO: increase the list
        if name == "elu":
            return nn.ELU()
        elif name == "selu":
            return nn.SELU()
        elif name == "relu":
            return nn.ReLU()
        elif name == "lrelu":
            return nn.LeakyReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "sigmoid":
            return nn.Sigmoid()
        return None

    def built_network_from_cfg(self, cfg: dict, state_space: gym.spaces.Space, action_space: gym.spaces.Space):
        # TODO: add output activation function
        layers = []
        hidden_layers = cfg.get("hidden_layers", [256, 256])
        activation = self.activation_funtion_by_name(cfg.get("hidden_activation", "relu"))

        # first layer
        layers.append(nn.Linear(*state_space.shape, hidden_layers[0]))
        layers.append(activation)

        # add remaining layers
        for i in range(len(hidden_layers)):
            # last layer
            if i == len(hidden_layers) - 1:
                layers.append(nn.Linear(hidden_layers[i], *action_space.shape))
            # hidden layers
            else:
                layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                layers.append(activation)

        # built network
        self.network = nn.Sequential(*layers)
