import argparse
import sys

from skrl import config, logger


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
    import functools
    from mujoco_playground import registry
    from mujoco_playground._src import wrapper

    import jax

    # check task from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--task"):
            defined = True
            break
    # get task name from command line arguments
    if defined:
        arg_index = sys.argv.index("--task") + 1
        if arg_index >= len(sys.argv):
            raise ValueError(
                "No task name defined. Set the task_name parameter or use --task <task_name> as command line argument"
            )
        if task_name and task_name != sys.argv[arg_index]:
            logger.warning(f"Overriding task ({task_name}) with command line argument ({sys.argv[arg_index]})")
    # get task name from function arguments
    else:
        if task_name:
            sys.argv.append("--task")
            sys.argv.append(task_name)
        else:
            raise ValueError(
                "No task name defined. Set the task_name parameter or use --task <task_name> as command line argument"
            )

    # check num_envs from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--num_envs"):
            defined = True
            break
    # get num_envs from command line arguments
    if defined:
        if num_envs is not None:
            logger.warning("Overriding num_envs with command line argument (--num_envs)")
    # get num_envs from function arguments
    elif num_envs is not None and num_envs > 0:
        sys.argv.append("--num_envs")
        sys.argv.append(str(num_envs))

    # parse arguments
    if parser is None:
        parser = argparse.ArgumentParser("MuJoCo Playground")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate")
    parser.add_argument("--task", type=str, default=None, help="Name of the task")
    # playground specific arguments
    parser.add_argument("--vision", action="store_true", default=False, help="Whether to use vision-based environment")

    args = parser.parse_args()
    task_name = args.task
    num_envs = args.num_envs
    vision = args.vision

    # randomization function
    randomizer = registry.get_domain_randomizer(task_name)
    if randomizer is not None:
        randomization_fn = functools.partial(randomizer, rng=jax.random.split(config.jax.key, num_envs))
    else:
        randomization_fn = None

    # load environment
    env_cfg = registry.get_default_config(task_name)
    env = wrapper.wrap_for_brax_training(
        registry.load(task_name, config=env_cfg),
        vision=vision,
        num_vision_envs=num_envs,
        episode_length=env_cfg.episode_length,
        action_repeat=1,
        randomization_fn=randomization_fn,
        full_reset=False,
    )

    # set number of environments
    env.num_envs = num_envs
    env.env.num_envs = num_envs
    env.env.unwrapped.num_envs = num_envs

    # print config
    if show_cfg:
        print(f"\nMuJoCo Playground environment ({task_name})")
        print(env_cfg)

    return env
