.. note::

    For models in JAX/Flax it is imperative to define all parameters (except ``observation_space``,
    ``state_space``, ``action_space`` and ``device``) with default values to avoid errors during initialization
    (``TypeError: __init__() missing N required positional argument``).

    In addition, it is necessary to initialize the model's ``state_dict`` (via the ``init_state_dict`` method) after
    its instantiation to avoid errors during its use (``AttributeError: object has no attribute "state_dict".
    If "state_dict" is defined in '.setup()', remember these fields are only accessible from inside 'init' or 'apply'``).
