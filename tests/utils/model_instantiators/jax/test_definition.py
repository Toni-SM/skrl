import yaml

import flax
import jax.numpy as jnp

from skrl.utils.model_instantiators.jax.common import _generate_modules, _get_activation_function, _parse_input


def test_get_activation_function(capsys):
    _globals = {"nn": flax.linen, "x": jnp.ones((1, 1))}

    activations = ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "softplus", "softsign", "selu", "softmax"]
    for item in activations:
        activation = _get_activation_function(item)
        assert activation is not None, f"{item} -> None"
        exec(f"{activation}(x)", _globals, {})


def test_parse_input(capsys):
    # check for tokens
    for input in ["STATES", "OBSERVATIONS", "ACTIONS", "OBSERVATION_SPACE", "STATE_SPACE", "ACTION_SPACE"]:
        output = _parse_input(input)
        for item in ["STATES", "OBSERVATIONS", "ACTIONS", "OBSERVATION_SPACE", "STATE_SPACE", "ACTION_SPACE"]:
            assert item not in output, f"'{item}' in '{output}'"
    # Mixed operation
    input = 'OBSERVATIONS["joint"] + concatenate([net * ACTIONS[:, -3:]]) - STATES["image"]'
    statement = 'observations["joint"] + jnp.concatenate([net * taken_actions[:, -3:]], axis=-1) - states["image"]'
    output = _parse_input(str(input))
    assert output.replace("'", '"') == statement, f"'{output}' != '{statement}'"


def test_generate_modules(capsys):
    _globals = {"nn": flax.linen}

    # activation functions
    content = r"""
    activations: [relu, tanh, sigmoid, leaky_relu, elu, softplus, softsign, selu, softmax]
    """
    content = yaml.safe_load(content)
    modules = _generate_modules([1] * len(content["activations"]), content["activations"])
    _locals = {}
    exec(f'container = nn.Sequential([{", ".join(modules)}])', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nactivations:", container)
    assert isinstance(container, flax.linen.Sequential)
    assert len(container.layers) == len(content["activations"]) * 2

    # linear
    content = r"""
    layers:
    - 8
    - linear: 16
    - linear: [32, True]
    - linear: {out_features: 64, bias: True}
    - linear: {features: 64, use_bias: False}
    activations: elu
    """
    content = yaml.safe_load(content)
    modules = _generate_modules(content["layers"], content["activations"])
    _locals = {}
    exec(f'container = nn.Sequential([{", ".join(modules)}])', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nlinear:", container)
    assert isinstance(container, flax.linen.Sequential)
    assert len(container.layers) == 10

    # conv2d
    content = r"""
    layers:
    - conv2d: [2, 4, [8, 16]]
    - conv2d: {out_channels: 16, kernel_size: 8, stride: [4, 2], padding: valid, bias: True}
    - conv2d: {features: 16, kernel_size: 8, strides: [4, 2], padding: VALID, use_bias: False}
    activations:
    - elu
    """
    content = yaml.safe_load(content)
    modules = _generate_modules(content["layers"], content["activations"])
    _locals = {}
    exec(f'container = nn.Sequential([{", ".join(modules)}])', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nconv2d:", container)
    assert isinstance(container, flax.linen.Sequential)
    assert len(container.layers) == 6

    # flatten
    content = r"""
    layers:
    - flatten
    - flatten: [2, -2]
    - flatten: {start_dim: 3, end_dim: -3}
    activations:
    - elu
    - elu
    - elu
    """
    content = yaml.safe_load(content)
    modules = _generate_modules(content["layers"], content["activations"])
    _locals = {}
    exec(f'container = nn.Sequential([{", ".join(modules)}])', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nflatten:", container)
    assert isinstance(container, flax.linen.Sequential)
    assert len(container.layers) == 3

    # non-lazy layers
    content = r"""
    layers:
    - linear: {in_features: OBSERVATIONS, out_features: ONE}
    - conv2d: {in_channels: 3, out_channels: 16, kernel_size: 8}
    """
    content = yaml.safe_load(content)
    modules = _generate_modules(content["layers"], content.get("activations", []))
    _locals = {}
    _globals["self"] = lambda: None
    _globals["self"].num_observations = 5
    exec(f'container = nn.Sequential([{", ".join(modules)}])', _globals, _locals)
    container = _locals["container"]
    with capsys.disabled():
        print("\nnon-lazy:", container)
    assert isinstance(container, flax.linen.Sequential)
    assert len(container.layers) == 2
