import argparse


def load_isaaclab_env(
    *,
    task_name: str = "",
    num_envs: int | None = None,
    headless: bool | None = None,
    cli_args: list[str] = [],
    show_cfg: bool = True,
    parser: argparse.ArgumentParser | None = None,
):
    """Load an Isaac Lab environment.

    Isaac Lab: https://isaac-sim.github.io/IsaacLab

    This function includes the definition and parsing of command line arguments used by Isaac Lab:

    - ``--num_envs``: Number of environments to simulate
    - ``--task``: Name of the task
    - ``--seed``: Seed used for the environment
    - ``--disable_fabric``: Disable fabric and use USD I/O operations.
    - ``--distributed``: Run training with multiple GPUs or nodes

    :param task_name: The name of the task.
        If not specified, the task name is taken from the command line argument (``--task TASK_NAME``).
        Command line argument has priority over function parameter if both are specified
    :param num_envs: Number of parallel environments to create.
        If not specified, the default number of environments defined in the task configuration is used.
        Command line argument has priority over function parameter if both are specified
    :param headless: Whether to use headless mode (no rendering).
        If not specified, the default task configuration is used.
        Command line argument has priority over function parameter if both are specified
    :param cli_args: Isaac Lab configuration and command line arguments.
    :param show_cfg: Whether to print the configuration.
    :param parser: The argument parser to use. If not specified, a new argument parser will be created.

    :return: Isaac Lab environment.

    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments.
    """
    # since Isaac Lab environments are implemented on top of PyTorch, the loader is the same
    from skrl.envs.loaders.torch import load_isaaclab_env

    return load_isaaclab_env(
        task_name=task_name, num_envs=num_envs, headless=headless, cli_args=cli_args, show_cfg=show_cfg, parser=parser
    )
