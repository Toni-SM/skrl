import argparse


def load_playground_env(
    *,
    task_name: str = "",
    num_envs: int | None = None,
    show_cfg: bool = True,
    parser: argparse.ArgumentParser | None = None,
):
    """Load a MuJoCo Playground environment.

    MuJoCo Playground: https://playground.mujoco.org

    This function includes the definition and parsing of command line arguments:

    - ``--num_envs``: Number of environments to simulate
    - ``--task``: Name of the task
    - ``--vision``: Whether to use vision-based environment

    :param task_name: The name of the task.
        If not specified, the task name is taken from the command line argument (``--task TASK_NAME``).
        Command line argument has priority over function parameter if both are specified
    :param num_envs: Number of parallel environments to create.
        If not specified, the default number of environments defined in the task configuration is used.
        Command line argument has priority over function parameter if both are specified
    :param show_cfg: Whether to print the configuration.
    :param parser: The argument parser to use. If not specified, a new argument parser will be created.

    :return: MuJoCo Playground environment.

    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments.
    """
    # since MuJoCo Playground environments are implemented on top of JAX, the loader is the same
    from skrl.envs.loaders.jax import load_playground_env

    return load_playground_env(task_name=task_name, num_envs=num_envs, show_cfg=show_cfg, parser=parser)
