from .module import Module, _Parameter


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        # register modules
        modules = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
        for i, module in enumerate(modules):
            if isinstance(module, _Parameter):
                raise ValueError("A parameter instance cannot be registered as a module in a Sequential container")
            self.register_module(str(i), module)

    def __len__(self):
        return len(self.modules())

    def forward(self, input):
        for module in self.modules():
            input = module(input)
        return input
