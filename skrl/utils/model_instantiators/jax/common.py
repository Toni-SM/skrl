from typing import Any, Mapping, Sequence, Tuple, Union

import ast

from skrl import logger


def _get_activation_function(activation: Union[str, None]) -> Union[str, None]:
    """Get the activation function

    Supported activation functions:

    - "elu"
    - "leaky_relu"
    - "relu"
    - "selu"
    - "sigmoid"
    - "softmax"
    - "softplus"
    - "softsign"
    - "tanh"

    :param activation: Activation function name

    :return: Activation function or None if the activation is not supported
    """
    activations = {
        "elu": "nn.elu",
        "leaky_relu": "nn.leaky_relu",
        "relu": "nn.relu",
        "selu": "nn.selu",
        "sigmoid": "nn.sigmoid",
        "softmax": "nn.softmax",
        "softplus": "nn.softplus",
        "softsign": "nn.soft_sign",
        "tanh": "nn.tanh",
    }
    return activations.get(activation.lower() if type(activation) is str else activation, None)

def _parse_input(source: str) -> str:
    """Parse a network input expression by replacing substitutions and applying operations

    :param source: Input expression

    :return: Parsed network input
    """
    class NodeTransformer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name):
                # operation: concatenate
                if node.func.id == "concatenate":
                    node.func = ast.Attribute(value=ast.Name("jnp"), attr="concatenate")
                    node.keywords = [ast.keyword(arg="axis", value=ast.Constant(value=-1))]
                # operation: permute
                if node.func.id == "permute":
                    node.func = ast.Attribute(value=ast.Name("jnp"), attr="permute_dims")
            return node

    # apply operations by modifying the source syntax grammar
    tree = ast.parse(source)
    NodeTransformer().visit(tree)
    source = ast.unparse(tree)
    # enum substitutions
    source = source.replace("Shape.STATES_ACTIONS", "STATES_ACTIONS").replace("STATES_ACTIONS", 'jnp.concatenate([inputs["states"], inputs["taken_actions"]], axis=-1)')
    source = source.replace("Shape.OBSERVATIONS_ACTIONS", "OBSERVATIONS_ACTIONS").replace("OBSERVATIONS_ACTIONS", 'jnp.concatenate([inputs["states"], inputs["taken_actions"]], axis=-1)')
    source = source.replace("Shape.STATES", "STATES").replace("STATES", 'inputs["states"]')
    source = source.replace("Shape.OBSERVATIONS", "OBSERVATIONS").replace("OBSERVATIONS", 'inputs["states"]')
    source = source.replace("Shape.ACTIONS", "ACTIONS").replace("ACTIONS", 'inputs["taken_actions"]')
    return source

def _parse_output(source: Union[str, Sequence[str]]) -> Tuple[Union[str, Sequence[str]], Sequence[str], int]:
    """Parse the network output expression by replacing substitutions and applying operations

    :param source: Output expression

    :return: Tuple with the parsed network output, generated modules and output size/shape
    """
    class NodeTransformer(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name):
                # operation: concatenate
                if node.func.id == "concatenate":
                    node.func = ast.Attribute(value=ast.Name("jnp"), attr="concatenate")
                    node.keywords = [ast.keyword(arg="axis", value=ast.Constant(value=-1))]
                # activation functions
                activation = _get_activation_function(node.func.id)
                if activation:
                    node.func = ast.Attribute(value=ast.Name("nn"), attr=activation.replace("nn.", ""))
            return node

    size = get_num_units("ACTIONS")
    modules = []
    if type(source) is str:
        # enum substitutions
        source = source.replace("Shape.ACTIONS", "ACTIONS").replace("Shape.ONE", "ONE")
        token = "ACTIONS" if "ACTIONS" in source else None
        token = "ONE" if "ONE" in source else token
        if token:
            size = get_num_units(token)
            modules = [f"nn.Dense(features={get_num_units(token)})"]
            source = source.replace(token, "PLACEHOLDER")
        # apply operations by modifying the source syntax grammar
        tree = ast.parse(source)
        NodeTransformer().visit(tree)
        source = ast.unparse(tree)
    elif type(source) in [list, tuple]:
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid or unsupported network output definition: {source}")
    return source, modules, size

def _generate_modules(layers: Sequence[str], activations: Union[Sequence[str], str]) -> Sequence[str]:
    """Generate network modules

    :param layers: Layer definitions
    :param activations: Activation function definitions applied after each layer (except ``flatten`` layers).
                        If a single activation function is specified (str or lis), it will be applied after each layer

    :return: A list of generated modules
    """
    # expand activations
    if type(activations) is str:
        activations = [activations] * len(layers)
    elif type(activations) is list:
        if not len(activations):
            activations = [""] * len(layers)
        elif len(activations) == 1:
            activations = activations * len(layers)
        elif len(activations) == len(layers):
            pass
        else:
            # TODO: check the length of activations
            raise NotImplementedError(f"Activations length ({len(activations)}) don't match layers ({len(layers)})")

    modules = []
    for layer, activation in zip(layers, activations):
        # single-values cases
        # linear (as number)
        if type(layer) in [int, float]:  # TODO: support token, e.g.: - ACTIONS??
            layer = {"linear": layer}
        # flatten (as string)
        elif type(layer) is str:
            layer = {"flatten": {}}

        # parse layer
        if type(layer) is dict:
            layer_type = next(iter(layer.keys())).lower()
            # linear
            if layer_type == "linear":
                cls = "nn.Dense"
                kwargs = layer[layer_type]
                if type(kwargs) in [int, float]:
                    kwargs = {"features": int(kwargs)}
                elif type(kwargs) is list:
                    kwargs = {k: v for k, v in zip(["features", "use_bias"][:len(kwargs)], kwargs)}
                elif type(kwargs) is dict:
                    if "in_features" in kwargs:
                        del kwargs["in_features"]
                    mapping = {
                        "out_features": "features",
                        "bias": "use_bias",
                    }
                    kwargs = {mapping.get(k, k): v for k, v in kwargs.items()}
                    kwargs["features"] = get_num_units(kwargs["features"])
                else:
                    raise ValueError(f"Invalid or unsupported 'linear' layer definition: {kwargs}")
            # convolutional 2D
            elif layer_type == "conv2d":
                cls = "nn.Conv"
                kwargs = layer[layer_type]
                if type(kwargs) is list:
                    kwargs = {k: v for k, v in zip(["features", "kernel_size", "strides", "padding", "use_bias"][:len(kwargs)], kwargs)}
                elif type(kwargs) is dict:
                    if "in_channels" in kwargs:
                        del kwargs["in_channels"]
                    mapping = {
                        "out_channels": "features",
                        "stride": "strides",
                        "bias": "use_bias",
                    }
                    kwargs = {mapping.get(k, k): f'"{v.upper()}"' if type(v) is str else v for k, v in kwargs.items()}
                else:
                    raise ValueError(f"Invalid or unsupported 'conv2d' layer definition: {kwargs}")
            # flatten
            elif layer_type == "flatten":
                cls = "lambda x: jnp.reshape(x, (x.shape[0], -1))"
                kwargs = None
                activation = ""  # don't add activation after flatten layer
            else:
                raise ValueError(f"Invalid or unsupported layer: {layer_type}")
        else:
            raise ValueError(f"Invalid or unsupported layer definition: {layer}")
        # define layer and activation function
        if kwargs is None:
            modules.append(f"{cls}")
        else:
            kwargs = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            modules.append(f"{cls}({kwargs})")
        activation = _get_activation_function(activation)
        if activation:
            modules.append(activation)
    return modules

def get_num_units(token: Union[str, Any]) -> Union[str, Any]:
    """Get the number of units/features a token represent

    :param token: Token

    :return: Number of units/features a token represent. If the token is unknown, its value will be returned as it
    """
    num_units = {
        "ONE": "1",
        "STATES": "self.num_observations",
        "OBSERVATIONS": "self.num_observations",
        "ACTIONS": "self.num_actions",
        "STATES_ACTIONS": "self.num_observations + self.num_actions",
        "OBSERVATIONS_ACTIONS": "self.num_observations + self.num_actions",
    }
    token_as_str = str(token).replace("Shape.", "")
    if token_as_str in num_units:
        return num_units[token_as_str]
    return token

def generate_containers(network: Sequence[Mapping[str, Any]], output: Union[str, Sequence[str]],
                        embed_output: bool = True, indent: int = -1) -> \
                        Tuple[Sequence[Mapping[str, Any]], Mapping[str, Any]]:
    """Generate network containers

    :param network: Network definition
    :param output: Network's output expression
    :param embed_output: Whether to embed the output modules (if any) in the container definition.
                         If True, the output modules will be append to the last container module
    :param indent: Indentation level used to generate the Sequential definition.
                   If negative, no indentation will be applied

    :return: Network containers and output
    """
    # parse output
    output, output_modules, output_size = _parse_output(output)
    # build containers
    containers = []
    for i, item in enumerate(network):
        container = {}
        container["name"] = item["name"]
        container["input"] = _parse_input(item["input"])
        container["modules"] = _generate_modules(item["layers"], item.get("activations", []))
        # embed output in the container definition
        if embed_output and i == len(network) - 1:
            container["modules"] += output_modules
            output_modules = []
        # define a Sequential container
        if indent < 0:
            container["sequential"] = f'nn.Sequential([{", ".join(container["modules"])}])'
        else:
            container["sequential"] = "nn.Sequential(["
            for item in container["modules"]:
                container["sequential"] += f"\n{' ' * 4 * indent}{item},"
            container["sequential"] += f"\n{' ' * 4 * (indent - 1)}])"
        containers.append(container)
    # compose output
    if type(output) is str:
        # avoid 'output = placeholder'
        if output == "PLACEHOLDER" or output == container["name"]:
            output = None
        # substitute placeholder in output expression
        else:
            output = output.replace("PLACEHOLDER", container["name"] if embed_output else "output")
    output = {"output": output, "modules": output_modules, "size": output_size}
    return containers, output

def convert_deprecated_parameters(parameters: Mapping[str, Any]) -> Tuple[Mapping[str, Any], str]:
    """Function to convert deprecated parameters to network-output format

    :param parameters: Deprecated parameters and their values.

    :return: Network and output definitions
    """
    logger.warning(f'The following parameters ({", ".join(list(parameters.keys()))}) are deprecated. '
                    "See https://skrl.readthedocs.io/en/latest/api/utils/model_instantiators.html")
    # network definition
    activations = parameters.get("hidden_activation", [])
    if type(activations) in [list, tuple] and len(set(activations)) == 1:
        activations = activations[0]
    network = [
        {
            "name": "net",
            "input": str(parameters.get("input_shape", "STATES")),
            "layers": parameters.get("hiddens", []),
            "activations": activations,
        }
    ]
    # output
    output_scale = parameters.get("output_scale", 1.0)
    scale_operation = f"{output_scale} * " if output_scale != 1.0 else ""
    if parameters.get("output_activation", None):
        output = f'{scale_operation}{parameters["output_activation"]}({str(parameters.get("output_shape", "ACTIONS"))})'
    else:
        output = f'{scale_operation}{str(parameters.get("output_shape", "ACTIONS"))}'

    return network, output
