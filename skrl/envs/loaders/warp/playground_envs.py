import argparse


def load_playground_env(
    *,
    task_name: str = "",
    num_envs: int | None = None,
    episode_length: int | None = None,
    action_repeat: int | None = None,
    full_reset: bool = False,
    randomization: bool = False,
    cfg_overrides: dict | None = None,
    show_cfg: bool = True,
    parser: argparse.ArgumentParser | None = None,
):
    """Load a MuJoCo Playground environment.

    MuJoCo Playground: https://playground.mujoco.org

    This function includes the definition and parsing of the following command line arguments:

    - ``--task``: Name of the task.
    - ``--num_envs``: Number of environments to simulate.
    - ``--seed``: Random seed.
    - ``--episode_length``: Length of the episode.
    - ``--action_repeat``: Number of times to repeat the given action per step.
    - ``--full_reset``: Whether to perform a full reset of the environment on each step, rather than resetting to an initial cached state.
    - ``--randomization``: Whether to use randomization.

    :param task_name: The name of the task.
        If not specified, the task name is taken from the command line argument (``--task TASK_NAME``).
        Command line argument has priority over function parameter if both are specified.
    :param num_envs: Number of parallel environments to create.
        If not specified, the number of environments is taken from the command line argument (``--num_envs N``).
        Command line argument has priority over function parameter if both are specified.
    :param episode_length: Length of the episode.
        If neither the function parameter nor the command line argument is specified, the default configuration for the task will be used.
        Command line argument has priority over function parameter if both are specified.
    :param action_repeat: Number of times to repeat the given action per step.
        If neither the function parameter nor the command line argument is specified, the default configuration for the task will be used.
        Command line argument has priority over function parameter if both are specified.
    :param full_reset: Whether to perform a full reset of the environment on each step, rather than resetting to an initial cached state.
        Enabling this option may increase wall clock time because it forces full resets to random states.
        Command line argument has priority over function parameter if both are specified.
    :param randomization: Whether to use randomization.
        If the environment does not provide a randomization function, the randomization flag is ignored.
        Command line argument has priority over function parameter if both are specified.
    :param cfg_overrides: Configuration overrides for the environment.
    :param show_cfg: Whether to print the configuration.
    :param parser: The argument parser to use. If not specified, a new argument parser will be created.

    :return: MuJoCo Playground environment.

    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments,
        or the task name is invalid.
    :raises ValueError: The number of environments has not been defined, neither by the function parameter nor by the command line arguments.
    :raises ValueError: The episode length has not been defined, neither by the function parameter nor by the command line arguments.
        The task configuration does not provide a default episode length.
    """
    # since MuJoCo Playground environments are implemented on top of JAX, the loader is the same
    from skrl.envs.loaders.jax import load_playground_env

    return load_playground_env(
        task_name=task_name,
        num_envs=num_envs,
        episode_length=episode_length,
        action_repeat=action_repeat,
        full_reset=full_reset,
        randomization=randomization,
        cfg_overrides=cfg_overrides,
        show_cfg=show_cfg,
        parser=parser,
    )
