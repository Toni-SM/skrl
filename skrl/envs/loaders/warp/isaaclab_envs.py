from __future__ import annotations

import argparse
from collections.abc import Sequence


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
    # since Isaac Lab environments are implemented on top of PyTorch, the loader is the same
    from skrl.envs.loaders.torch import load_isaaclab_env

    return load_isaaclab_env(
        task_name=task_name,
        num_envs=num_envs,
        physics=physics,
        renderer=renderer,
        presets=presets,
        visualizer=visualizer,
        cli_args=cli_args,
        show_cfg=show_cfg,
        parser=parser,
    )
