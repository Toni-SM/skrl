from typing import Any, Mapping, Optional, Sequence, Union

import ast
from enum import Enum


class Shape(Enum):
    """
    Enum to select the shape of the model's inputs and outputs
    """
    ONE = 1
    STATES = 0
    OBSERVATIONS = 0
    ACTIONS = -1
    STATES_ACTIONS = -2
    OBSERVATIONS_ACTIONS = -2

ACTIVATIONS = {
    "relu": "nn.ReLU()",
    "tanh": "nn.Tanh()",
    "sigmoid": "nn.Sigmoid()",
    "leaky_relu": "nn.LeakyReLU()",
    "elu": "nn.ELU()",
    "softplus": "nn.Softplus()",
    "softsign": "nn.Softsign()",
    "selu": "nn.SELU()",
    "softmax": "nn.Softmax()",
}


def _parse_input(source: str) -> str:
    class NodeTransformer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name):
                # operation: concatenate
                if node.func.id == "concatenate":
                    node.func = ast.Attribute(value=ast.Name("torch"), attr="cat")
                    node.keywords = [ast.keyword(arg="dim", value=ast.Constant(value=1))]
            return node
    # apply operations
    tree = ast.parse(source)
    NodeTransformer().visit(tree)
    source = ast.unparse(tree)
    # Shape enum
    source = source.replace("Shape.STATES_ACTIONS", "STATES_ACTIONS").replace("STATES_ACTIONS", 'torch.cat((inputs["states"], inputs["taken_actions"]), dim=1)')
    source = source.replace("Shape.OBSERVATIONS_ACTIONS", "OBSERVATIONS_ACTIONS").replace("OBSERVATIONS_ACTIONS", 'torch.cat((inputs["states"], inputs["taken_actions"]), dim=1)')
    source = source.replace("Shape.STATES", "STATES").replace("STATES", 'inputs["states"]')
    source = source.replace("Shape.OBSERVATIONS", "OBSERVATIONS").replace("OBSERVATIONS", 'inputs["states"]')
    source = source.replace("Shape.ACTIONS", "ACTIONS").replace("ACTIONS", 'inputs["taken_actions"]')
    return source

def _generate_sequential(layers, activations) -> str:
    def _num_units(shape: Union[Shape, str, Any]) -> Union[str, Any]:
        num_units = {
            "ONE": "1",
            "STATES": "self.num_observations",
            "OBSERVATIONS": "self.num_observations",
            "ACTIONS": "self.num_actions",
            "STATES_ACTIONS": "self.num_observations + self.num_actions",
            "OBSERVATIONS_ACTIONS": "self.num_observations + self.num_actions",
        }
        shape_as_str = str(shape).replace("Shape.", "")
        if shape_as_str in num_units:
            return num_units[shape_as_str]
        return shape

    # expand activations
    if type(activations) is str:
        activations = [activations] * len(layers)
    elif type(activations) is list:
        if not len(activations):
            activations = [""] * len(layers)
        elif len(activations) == 1:
            activations = activations * len(layers)
        else:
            pass # TODO: check the length of activations

    modules = []
    for layer, activation in zip(layers, activations):
        # special cases
        # linear (as int)
        if type(layer) in [int, float]:
            layer = {"linear": layer}
        # flatten (without value)
        elif type(layer) is str:
            layer = {"flatten": {}}

        # parse layer
        if type(layer) is dict:
            layer_type = next(iter(layer.keys())).lower()
            # linear
            if layer_type == "linear":
                cls = "nn.LazyLinear"
                kwargs = layer[layer_type]
                if type(kwargs) in [int, float]:
                    kwargs = {"out_features": int(kwargs)}
                elif type(kwargs) is list:
                    kwargs = {k: v for k, v in zip(["out_features", "bias"][:len(kwargs)], kwargs)}
                elif type(kwargs) is dict:
                    mapping = {
                        "features": "out_features",
                        "use_bias": "bias",
                    }
                    kwargs = {mapping.get(k, k): v for k, v in kwargs.items()}
                    kwargs["out_features"] = _num_units(kwargs["out_features"])
                    # non-lazy module
                    if "in_features" in kwargs:
                        cls = "nn.Linear"
                        kwargs["in_features"] = _num_units(kwargs["in_features"])
                else:
                    raise ValueError(f"Invalid or unsupported 'linear' layer definition: {kwargs}")
            # convolutional 2D
            elif layer_type == "conv2d":
                cls = "nn.LazyConv2d"
                kwargs = layer[layer_type]
                if type(kwargs) is list:
                    kwargs = {k: v for k, v in zip(["out_channels", "kernel_size", "stride", "padding", "bias"][:len(kwargs)], kwargs)}
                elif type(kwargs) is dict:
                    mapping = {
                        "features": "out_channels",
                        "strides": "stride",
                        "use_bias": "bias",
                    }
                    kwargs = {mapping.get(k, k): f'"{v.lower()}"' if type(v) is str else v for k, v in kwargs.items()}
                    # non-lazy module
                    if "in_channels" in kwargs:
                        cls = "nn.Conv2d"
                else:
                    raise ValueError(f"Invalid or unsupported 'conv2d' layer definition: {kwargs}")
            # flatten
            elif layer_type == "flatten":
                cls = "nn.Flatten"
                activation = ""  # don't add activation after flatten layer
                kwargs = layer[layer_type]
                if type(kwargs) is list:
                    kwargs = {k: v for k, v in zip(["start_dim", "end_dim"][:len(kwargs)], kwargs)}
                elif type(kwargs) is dict:
                    pass
                else:
                    raise ValueError(f"Invalid or unsupported 'flatten' layer definition: {kwargs}")
            else:
                raise ValueError(f"Invalid or unsupported layer: {layer_type}")
        else:
            raise ValueError(f"Invalid or unsupported layer definition: {layer}")
        # define layer and activation function
        kwargs = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        modules.append(f"{cls}({kwargs})")
        if activation.lower() in ACTIVATIONS:
            modules.append(ACTIVATIONS[activation.lower()])
    return modules

def generate_containers(network: Sequence[Mapping[str, Any]]) -> Sequence[str]:
    containers = []
    for item in network:
        container = {}
        container["name"] = item["name"]
        container["input"] = _parse_input(item["input"])
        container["sequential"] = _generate_sequential(item["layers"], item.get("activations", []))
        containers.append(container)
    return containers
