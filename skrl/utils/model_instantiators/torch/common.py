from typing import Optional, Union, Sequence, Mapping, Any

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
    # expand activations
    if type(activations) is str:
        activations = [activations] * len(layers)
    
    modules = []
    for layer in layers:
        if type(layer) is dict:
            if next(iter(layer.keys())).lower() == "linear":
                kwargs = layer["linear"]
                if type(kwargs) in [int, float]:
                    kwargs = {"out_features": int(kwargs)}
                if type(kwargs) is list:
                    kwargs = {k: v for k, v in zip()}
                args = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                modules.append(f"nn.LazyLinear({args})")
        pass

def generate_containers(network: Sequence[Mapping[str, Any]]) -> Sequence[str]:
    containers = []
    for item in network:
        container = {}
        container["name"] = item["name"]
        container["input"] = _parse_input(item["input"])
        container["sequential"] = _generate_sequential(item["layers"], item["activations"])
        containers.append(container)
    return containers
