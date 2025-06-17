from typing import Iterator, Mapping, Optional, Sequence, Tuple

from abc import ABC
from collections import OrderedDict

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
