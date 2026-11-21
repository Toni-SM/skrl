from __future__ import annotations

from typing import Any

import argparse

from skrl import config


def load_playground_env(
    *,
    task_name: str = "",
    num_envs: int | None = None,
    device: str | None = None,
    cfg_overrides: dict[str, Any] | None = None,
    show_cfg: bool = True,
    parser: argparse.ArgumentParser | None = None,
):
    """Load a MuJoCo Playground environment.

    MuJoCo Playground: https://playground.mujoco.org

    .. note::

        MuJoCo Playground 1.0.0 or later (implemented on top of PyTorch and MuJoCo Warp) is required.

    This function includes the definition and parsing of the following command line arguments:

    - ``--task``: Name of the task.
    - ``--num_envs``: Number of environments to simulate.
    - ``--device``: Device (e.g.: ``"cpu"``, ``"cuda:0"``) used to simulate the environments.
    - ``--seed``: Random seed.

    :param task_name: The name of the task.
        If not specified, the task name is taken from the command line argument (``--task TASK_NAME``).
        Command line argument has priority over function parameter if both are specified.
    :param num_envs: Number of parallel environments to create.
        If not specified, the number of environments is taken from the command line argument (``--num_envs N``).
        Command line argument has priority over function parameter if both are specified.
    :param device: Device used to simulate the environments.
        If neither the function parameter nor the command line argument (``--device DEVICE``) is specified,
        skrl's default Warp device (``skrl.config.warp.device``) is used.
        Command line argument has priority over function parameter if both are specified.
    :param cfg_overrides: Task configuration overrides, where the keys are (dotted) attribute paths of the
        task configuration (e.g.: ``{"episode_length_secs": 5.0, "sim.impl": "cpp"}``).
        The ``"sim.nworld"`` and ``"sim.xpu_device"`` keys are not allowed (use ``num_envs`` and ``device`` instead).
    :param show_cfg: Whether to print the configuration.
    :param parser: The argument parser to use. If not specified, a new argument parser will be created.

    :return: MuJoCo Playground environment.

    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments,
        or the task name is invalid.
    :raises ValueError: The number of environments has not been defined, neither by the function parameter nor by the command line arguments.
    :raises ValueError: A configuration override sets the number of environments or the device.
    :raises AttributeError: A configuration override does not exist in the task configuration.
    """
    # since MuJoCo Playground environments are implemented on top of PyTorch, the loader is the same
    from skrl.envs.loaders.torch import load_playground_env

    if device is None:
        device = config.warp.device.alias

    return load_playground_env(
        task_name=task_name,
        num_envs=num_envs,
        device=device,
        cfg_overrides=cfg_overrides,
        show_cfg=show_cfg,
        parser=parser,
    )
