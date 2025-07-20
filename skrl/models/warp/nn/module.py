from typing import Iterator, Mapping, Optional, Sequence, Tuple

from abc import ABC
from collections import OrderedDict

import numpy as np
import warp as wp


class Module(ABC):
    _parameters: Mapping[str, Optional[wp.array]]
    _modules: Mapping[str, Optional["Module"]]

    def __init__(self, *args, **kwargs):
        self.__dict__["_attrs"] = OrderedDict()
        super().__setattr__("_parameters", OrderedDict())
        super().__setattr__("_modules", OrderedDict())

        self.device = wp.get_device("cuda")

    def __setattr__(self, key, value):
        self._attrs[key] = value

    def __getattr__(self, key):
        return self._attrs[key]

    def __post_init__(self) -> None:
        for k, v in self._attrs.items():
            if isinstance(v, Module):
                self.register_module(k, v)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _save_to_state_dict(self, destination, prefix):
        for name, param in self._parameters.items():
            if param is not None:
                destination[prefix + name] = param

    def forward(self, *args):
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "forward" function')

    def register_parameter(self, name: str, param: Optional[wp.array]) -> None:
        self._parameters[name] = param

    def register_module(self, name: str, module: Optional["Module"]) -> None:
        if not isinstance(module, Module):
            raise TypeError(f"{type(module)} is not a Module subclass")
        if name in self._modules:
            raise KeyError(f"name '{name}' already exists")
        self._modules[name] = module

    def parameters(self) -> Iterator[Optional[wp.array]]:
        modules = self._modules.values()
        if modules:
            parameters = []
            for module in modules:
                parameters += module.parameters()
        else:
            parameters = self._parameters.values()
        return parameters

    def modules(self) -> Iterator["Module"]:
        return self._modules.values()

    def named_modules(self) -> Iterator[Tuple[str, "Module"]]:
        return self._modules.items()

    def parse(self, uid: str) -> Tuple[str, Sequence[str], Sequence[str], Sequence[str], Sequence[str]]:
        raise NotImplementedError(f'Module [{type(self).__name__}] is missing the required "parse" function')

    def state_dict(
        self, *, destination: Optional[Mapping[str, wp.array]] = None, prefix: str = ""
    ) -> Mapping[str, wp.array]:
        if destination is None:
            destination = OrderedDict()
        self._save_to_state_dict(destination, prefix)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination=destination, prefix=prefix + name + ".")
        return destination

    def load_state_dict(self, state_dict: Mapping[str, wp.array]) -> None:
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
