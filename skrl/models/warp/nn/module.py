from __future__ import annotations

from abc import ABC
from collections import OrderedDict

import numpy as np
import warp as wp

from skrl import config


class _Parameter(ABC):
    pass


class Module(ABC):
    def __init__(self, *args, **kwargs):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

        self.device = config.warp.device

    def __post_init__(self) -> None:
        for k, v in self.__dict__.items():
            if isinstance(v, _Parameter):
                self.register_parameter(k, v)
            elif isinstance(v, Module):
                self.register_module(k, v)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _save_to_state_dict(self, destination, prefix):
        for name, parameter in self._parameters.items():
            if parameter is not None:
                destination[prefix + name] = parameter.data

    def forward(self, *args):
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" method')

    def register_parameter(self, name: str, parameter: _Parameter) -> None:
        if not isinstance(parameter, _Parameter):
            raise TypeError(f"Class {type(parameter)} is not a Parameter subclass")
        self._parameters[name] = parameter

    def register_module(self, name: str, module: Module) -> None:
        if not isinstance(module, Module):
            raise TypeError(f"Class {type(module)} is not a Module subclass")
        if name in self._modules:
            raise KeyError(f"Module with name '{name}' already exists")
        self._modules[name] = module

    def parameters(self, *, as_array: bool = True) -> list[_Parameter | wp.array]:
        modules = self._modules.values()
        if modules:
            parameters = []
            for module in modules:
                parameters += [
                    p.data if isinstance(p, _Parameter) and as_array else p
                    for p in module.parameters(as_array=as_array)
                ]
        else:
            parameters = [p.data if as_array else p for p in self._parameters.values()]
        return parameters

    def named_parameters(self) -> tuple[str, _Parameter]:
        return self._parameters.items()

    def modules(self) -> list[Module]:
        return self._modules.values()

    def named_modules(self) -> tuple[str, Module]:
        return self._modules.items()

    def state_dict(self, *, destination: dict[str, wp.array] | None = None, prefix: str = "") -> dict[str, wp.array]:
        if destination is None:
            destination = OrderedDict()
        self._save_to_state_dict(destination, prefix)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + ".")
        return destination

    def load_state_dict(self, state_dict: dict[str, wp.array]) -> None:
        def _load_from_state_dict(dst, src):
            if isinstance(src, dict):
                for k in src:
                    _load_from_state_dict(dst[k], src[k])
            elif isinstance(src, wp.array):
                wp.copy(dst, src.to(dst.device))
            elif isinstance(src, np.ndarray):
                wp.copy(dst, wp.array(src, dtype=dst.dtype, device=dst.device))
            else:
                raise NotImplementedError(f"Unsupported type: {type(src)}")

        _load_from_state_dict(self.state_dict(), state_dict)

    def to(self, device: wp.context.Device) -> None:
        self.device = config.warp.parse_device(device)
        for module in self._modules.values():
            module.to(self.device)
        for parameter in self._parameters.values():
            parameter.to(self.device)
        if isinstance(self, _Parameter):
            self.data = self.data.to(self.device)
