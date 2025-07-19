from typing import Iterator, Sequence, Tuple

import hashlib
import importlib.util
import os
import textwrap
import jinja2

import warp as wp


TEMPLATE_SEQUENTIAL = """
import warp as wp

TILE_THREADS = {{ tile_threads }}

{% for function in functions %}
{{ function }}\n
{% endfor %}

def create_kernel(input_size: int, output_size: int):
    @wp.kernel
    def forward(
        input: wp.array2d(dtype=float),
        out: wp.array2d(dtype=float),
    {% for kernel_argument in kernel_arguments %}
        {{ kernel_argument }},
    {% endfor %}
    ):
        i = wp.tid()

        # Load input
        x = wp.tile_load(input, shape=(TILE_THREADS, input_size), offset=(i * TILE_THREADS, 0))
        x = wp.tile_transpose(x)

        {% for kernel_definition in kernel_definitions %}
        {{ kernel_definition }}
        {% endfor %}

        # Store output
        wp.tile_store(out, wp.tile_transpose(x), offset=(i * TILE_THREADS, 0))

    return forward
"""


def parse_modules(
    named_modules: Iterator[Tuple[str, "Module"]], uid: str, input: str, output: str
) -> Tuple[str, Sequence[str], Sequence[str], Sequence[str], Sequence[str]]:
    from .activations import ReLU, Tanh
    from .linear import Linear
    from .sequential import Sequential

    functions = []
    kernel_parameters = []
    kernel_arguments = []
    kernel_definitions = []

    for i, (name, module) in enumerate(named_modules):
        # activation functions
        if isinstance(module, (ReLU, Tanh)):
            _uid = f"{uid}_fun{i}"
            _, _functions, _kernel_parameters, _kernel_arguments, _kernel_definitions = module.parse(_uid)
            functions.extend(_functions)
            kernel_parameters.extend(_kernel_parameters)
            kernel_arguments.extend(_kernel_arguments)
            kernel_definitions.extend([item.format(input="x", output="x") for item in _kernel_definitions])
        # layers
        # - linear
        elif isinstance(module, Linear):
            _uid = f"{uid}_lin{i}"
            _, _functions, _kernel_parameters, _kernel_arguments, _kernel_definitions = module.parse(_uid)
            functions.extend(_functions)
            kernel_parameters.extend(_kernel_parameters)
            kernel_arguments.extend(_kernel_arguments)
            kernel_definitions.extend([item.format(input="x", output="x") for item in _kernel_definitions])
        # containers
        # - sequential
        elif isinstance(module, Sequential):
            _uid = f"{uid}_seq{i}"
            _, _functions, _kernel_parameters, _kernel_arguments, _kernel_definitions = parse_modules(
                module.named_modules(), _uid
            )
            functions.extend(_functions)
            kernel_parameters.extend(_kernel_parameters)
            kernel_arguments.extend(_kernel_arguments)
            kernel_definitions.extend([item.format(input="x", output="x") for item in _kernel_definitions])

    return None, functions, kernel_parameters, kernel_arguments, kernel_definitions


def sequential_kernel_factory(module: "Sequential", tile_threads: int):
    from .sequential import Sequential

    if not isinstance(module, Sequential):
        raise ValueError(f"Module ({type(module)}) is not a Sequential module")

    # get source code and parameters
    _, functions, kernel_parameters, kernel_arguments, kernel_definitions = parse_modules(
        module.named_modules(), "", None, None
    )
    functions = sorted(set(functions))
    kernel_definitions = [textwrap.indent(item, prefix=" " * 8)[8:] for item in kernel_definitions]

    # render source code
    template = jinja2.Template(
        TEMPLATE_SEQUENTIAL,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    source = template.render(
        tile_threads=tile_threads,
        functions=functions,
        kernel_arguments=kernel_arguments,
        kernel_definitions=kernel_definitions,
    )

    # write source code to file
    hash = hashlib.sha256(source.encode()).hexdigest()[:8]
    cache_dir = os.path.join(wp.config.kernel_cache_dir, "nn")
    file_path = os.path.join(cache_dir, f"sequential_kernel_{hash}.py")
    os.makedirs(cache_dir, exist_ok=True)
    with open(file_path, "w") as file:
        file.write(source)

    # load source code from module
    spec = importlib.util.spec_from_file_location("sequential_kernel", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return source, module, kernel_parameters


def module_kernel_factory(module: "Module", tile_threads: int):
    import ast
    import inspect
    import textwrap

    raise NotImplementedError("Not implemented yet")

    # get 'forward' source code
    source = inspect.getsource(module.forward)
    source = textwrap.dedent(source)
    tree = ast.parse(source)

    # iterate over modules
    for i, _module in enumerate(list(module.modules())):
        pass
