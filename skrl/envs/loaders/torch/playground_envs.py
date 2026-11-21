from __future__ import annotations

from typing import Any

import argparse
import dataclasses

from skrl import config
from skrl.utils import set_seed


def _apply_overrides(cfg: Any, overrides: dict[str, Any]) -> None:
    """Apply (in-place) dotted-key overrides to a (nested) configuration object.

    :param cfg: The configuration object (e.g.: dataclass instance).
    :param overrides: The overrides, where the keys are (dotted) attribute paths (e.g.: ``"sim.impl"``).

    :raises AttributeError: The attribute path does not exist in the configuration object.
    """
    for key, value in overrides.items():
        *path, leaf = key.split(".")
        obj = cfg
        for name in path:
            obj = getattr(obj, name)
        if not hasattr(obj, leaf):
            raise AttributeError(f"Invalid configuration override: '{key}' ('{leaf}' attribute not found)")
        setattr(obj, leaf, value)


def _resolve_device(device: str | None) -> str:
    """Resolve the device specification to the string expected by MuJoCo Playground (e.g.: ``"cuda:0"``).

    :param device: Device specification (e.g.: ``"cpu"``, ``"cuda"``, ``"cuda:1"``).
        If not specified, skrl's default PyTorch device is used if CUDA is available, ``"cpu"`` otherwise.
        Invalid specifications fall back to the default available device.

    :return: ``"cpu"`` or ``"cuda:<index>"`` (MuJoCo Playground requires an explicit CUDA index).
    """
    import torch

    if not device and torch.cuda.is_available():
        device = str(config.torch.device)  # the default device is not validated (it may be CUDA when not available)
    _device = config.torch.parse_device(device)
    return f"cuda:{_device.index or 0}" if _device.type == "cuda" else _device.type


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
        skrl's default PyTorch device (``skrl.config.torch.device``) is used.
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
    from mujoco_playground.rl import env_registry
    from reaf.simulation.torch import flat_scene

    # parse arguments
    if parser is None:
        parser = argparse.ArgumentParser("MuJoCo Playground")
    parser.add_argument("--task", type=str, default=None, help="Name of the task")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate")
    parser.add_argument("--device", type=str, default=None, help="Device used to simulate the environments")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args, _ = parser.parse_known_args()

    task_name = args.task or task_name
    if not task_name:
        raise ValueError(
            "No task name defined. Set the 'task_name' parameter or use '--task <task_name>' as command line argument"
        )
    entry = env_registry.get(task_name)  # raises ValueError (listing the available tasks) if the task is invalid
    num_envs = args.num_envs or num_envs
    if num_envs is None:
        raise ValueError(
            "No number of environments defined. Set the 'num_envs' parameter or use '--num_envs <num_envs>' as command line argument"
        )
    device = _resolve_device(args.device or device)

    # adjust environment configuration
    env_cfg = entry.env_cfg_fn()
    reserved_overrides = {"sim.nworld", "sim.xpu_device"}.intersection(cfg_overrides or {})
    if reserved_overrides:
        raise ValueError(
            f"Invalid configuration override(s): {', '.join(sorted(reserved_overrides))}. "
            "Use the 'num_envs' and 'device' parameters (or command line arguments) instead"
        )
    _apply_overrides(env_cfg, cfg_overrides or {})
    env_cfg.sim.nworld = num_envs
    env_cfg.sim.xpu_device = device

    # print config
    if show_cfg:
        print(f"\nMuJoCo Playground environment")
        print(f"  task: {task_name}")
        print(f"  task config:")
        print(f"    episode_length_secs: {env_cfg.episode_length_secs}")
        if isinstance(env_cfg.observation_adapter, dict):
            print(f"    observation groups: {', '.join(env_cfg.observation_adapter)}")
        print(f"    sim:")
        for field in dataclasses.fields(env_cfg.sim):
            print(f"      {field.name}: {getattr(env_cfg.sim, field.name)}")
        print(f"  loading config:")
        print(f"    num_envs: {num_envs}")
        print(f"    device: {device}")
        print(f"    cfg_overrides: {cfg_overrides}")

    # set seed before building the environment (the environment build/startup may sample random numbers)
    set_seed(args.seed)
    return flat_scene.build_env(env_cfg)
