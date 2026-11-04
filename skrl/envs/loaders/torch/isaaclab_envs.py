from __future__ import annotations

import argparse
import atexit
import contextlib
import sys
from collections.abc import Sequence
import gymnasium

from skrl import logger


__all__ = ["load_isaaclab_env"]


def _print_cfg(d, indent=0) -> None:
    """Print the environment configuration.

    :param d: The dictionary to print.
    :param indent: The indentation level.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            print("  |   " * indent + f"  |-- {key}:")
            _print_cfg(value, indent + 1)
        else:
            print("  |   " * indent + f"  |-- {key}: {value}")


def _join(value: str | Sequence[str] | None) -> str | None:
    """Join a sequence of names into a comma-separated string.

    :param value: A name, a sequence of names, or None.

    :return: The comma-separated names, or None if no names were given.
    """
    if value is None or isinstance(value, str):
        return value or None
    return ",".join(value) or None


def _set_cli_option(flags: Sequence[str], value: str | None) -> None:
    """Append a command line option to ``sys.argv``, unless it is already defined there.

    The command line argument has priority over the given value.

    :param flags: The option flags (e.g.: ``["--visualizer", "--viz"]``). The first one is used when appending.
    :param value: The option value. If None, nothing is appended.
    """
    prefixes = tuple(f"{flag}=" for flag in flags)
    if any(arg in flags or arg.startswith(prefixes) for arg in sys.argv[1:]):
        if value is not None:
            logger.warning(f"Overriding {flags[0]} ({value}) with command line argument")
    elif value is not None:
        sys.argv += [flags[0], value]


def _set_cli_override(key: str, value: str | None) -> None:
    """Append a Hydra-style ``key=value`` override to ``sys.argv``, unless it is already defined there.

    The command line argument has priority over the given value.

    :param key: The override key (e.g.: ``"physics"``).
    :param value: The override value. If None, nothing is appended.
    """
    if any(arg.split("=", 1)[0].lstrip("-") == key for arg in sys.argv[1:] if "=" in arg):
        if value is not None:
            logger.warning(f"Overriding {key} ({value}) with command line argument")
    elif value is not None:
        sys.argv.append(f"{key}={value}")


def _request_determinism(physics_cfg) -> None:
    """Request deterministic physics and Warp kernels (as done by Isaac Lab's training scripts).

    Newton's sensor and geometry modules (e.g.: scene BVH used by tiled cameras) do not honor the physics
    configuration's ``deterministic`` flag but Warp's global ``warp.config.deterministic`` setting instead.
    This setting is only raised, never weakened.

    :param physics_cfg: Resolved physics configuration, or None if the configuration has none.
    """
    import warp as wp

    if physics_cfg is not None:
        physics_cfg.deterministic = True
    requested = getattr(physics_cfg, "deterministic_mode", None)
    mode = getattr(wp.DeterministicMode, requested.upper(), None) if isinstance(requested, str) else None
    if mode is None or mode == wp.DeterministicMode.NOT_GUARANTEED:
        mode = wp.DeterministicMode.RUN_TO_RUN
    if wp.config.deterministic < mode:
        wp.config.deterministic = mode


def load_isaaclab_env(
    *,
    task_name: str = "",
    num_envs: int | None = None,
    physics: str | None = None,
    renderer: str | None = None,
    presets: str | Sequence[str] | None = None,
    visualizer: str | Sequence[str] | None = None,
    cli_args: Sequence[str] = (),
    show_cfg: bool = True,
    parser: argparse.ArgumentParser | None = None,
):
    """Load an Isaac Lab environment.

    Isaac Lab: https://isaac-sim.github.io/IsaacLab

    This function includes the definition and parsing of command line arguments used by Isaac Lab:

    - ``--num_envs``: Number of environments to simulate
    - ``--task``: Name of the task
    - ``--seed``: Seed used for the environment
    - ``--disable_fabric``: Disable fabric and use USD I/O operations
    - ``--distributed``: Run training with multiple GPUs or nodes
    - Isaac Lab simulation launcher arguments (e.g.: ``--device``, ``--visualizer``/``--viz``, ``--deterministic``)
    - Isaac Lab Hydra-style backend/preset selectors and overrides (e.g.: ``physics=newton_mjwarp``,
      ``renderer=newton_renderer``, ``presets=NAME[,NAME,...]``, ``env.scene.num_envs=64``)

    The Isaac Sim (Kit) runtime is launched only if the resolved configuration requires it
    (e.g.: ``physics=isaacsim_physx``, ``renderer=isaacsim_rtx`` or ``--viz kit``).

    :param task_name: The name of the task.
        If not specified, the task name is taken from the command line argument (``--task TASK_NAME``).
        Command line argument has priority over function parameter if both are specified
    :param num_envs: Number of parallel environments to create.
        If not specified, the default number of environments defined in the task configuration is used.
        Command line argument has priority over function parameter if both are specified
    :param physics: Physics backend
        (e.g.: ``"newton_mjwarp"``, ``"newton_kamino"``, ``"ovphysx"``, ``"isaacsim_physx"``).
        If not specified, the default task configuration is used.
        Command line argument (``physics=NAME``) has priority over function parameter if both are specified
    :param renderer: Rendering backend for camera-based tasks
        (e.g.: ``"newton_renderer"``, ``"ovrtx"``, ``"isaacsim_rtx"``, ``"rtx"``).
        If not specified, the default task configuration is used.
        Command line argument (``renderer=NAME``) has priority over function parameter if both are specified
    :param presets: Task-specific configuration preset name(s).
        If not specified, the default task configuration is used.
        Unlike other parameters, they are merged with the command line presets (``presets=NAME[,NAME,...]``).
        Isaac Lab raises an error if the merged presets select conflicting alternatives
    :param visualizer: Visualizer backend(s) to run (e.g.: ``"kit"``, ``"newton_gl"``, ``"rerun"``, ``"viser"``).
        If not specified, the simulation runs headless.
        Command line argument (``--visualizer``/``--viz``) has priority over function parameter if both are specified
    :param cli_args: Isaac Lab configuration and command line arguments (e.g.: ``["--seed", "42", "physics=ovphysx"]``).
        They are treated as if they were specified on the command line.
    :param show_cfg: Whether to print the configuration.
    :param parser: The argument parser to use. If not specified, a new argument parser will be created.

    :return: Isaac Lab environment.

    :raises TypeError: The ``cli_args`` parameter is a string instead of a sequence of strings.
    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments.
    """
    if isinstance(cli_args, str):
        raise TypeError(f"cli_args must be a sequence of strings, not a string ({cli_args!r})")
    sys.argv += cli_args

    # merge function parameters into the command line arguments (which have priority)
    _set_cli_option(["--task"], task_name or None)
    _set_cli_option(["--num_envs"], str(num_envs) if num_envs is not None and num_envs > 0 else None)
    _set_cli_option(["--visualizer", "--viz"], _join(visualizer))
    _set_cli_override("physics", physics)
    _set_cli_override("renderer", renderer)
    # presets are additive (Isaac Lab accumulates all the presets=... tokens)
    if _join(presets):
        sys.argv.append(f"presets={_join(presets)}")

    # parse arguments
    if parser is None:
        parser = argparse.ArgumentParser("Isaac Lab")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate")
    parser.add_argument("--task", type=str, default=None, help="Name of the task")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations"
    )
    parser.add_argument(
        "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes"
    )

    import isaaclab_tasks  # type: ignore
    from isaaclab.app import add_launcher_args, launch_simulation  # type: ignore
    from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli  # type: ignore

    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # type: ignore

    add_launcher_args(parser)
    # the remaining arguments are the Hydra-style backend/preset selectors and overrides
    args, hydra_args = setup_preset_cli(parser)
    if not args.task:
        raise ValueError(
            "No task name defined. Set the task_name parameter or use --task <task_name> as command line argument"
        )

    # resolve the environment configuration (safe to call before launching the simulation)
    cfg, _ = resolve_task_config(args.task, None, overrides=hydra_args)

    # launch the simulation (Isaac Sim is launched only if the configuration requires it).
    # on success, the simulation is kept alive until the Python interpreter exits
    with contextlib.ExitStack() as stack:
        stack.enter_context(launch_simulation(cfg, args))

        if args.num_envs is not None:
            cfg.scene.num_envs = args.num_envs
        if args.seed is not None:
            cfg.seed = args.seed
        if args.disable_fabric:
            cfg.sim.use_fabric = False
        if getattr(args, "deterministic", False):
            _request_determinism(getattr(cfg.sim, "physics", None))

        # print config
        if show_cfg:
            print(f"\nIsaac Lab environment ({args.task})")
            _print_cfg(cfg.to_dict())

        # load environment
        env = gymnasium.make(args.task, cfg=cfg)

        atexit.register(stack.pop_all().close)

    return env
