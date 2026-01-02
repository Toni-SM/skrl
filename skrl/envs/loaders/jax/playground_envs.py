import argparse
import textwrap

from skrl import config
from skrl.utils import set_seed


def load_playground_env(
    *,
    task_name: str = "",
    num_envs: int | None = None,
    episode_length: int | None = None,
    action_repeat: int | None = None,
    full_reset: bool = False,
    randomization: bool = False,
    vision: bool = False,
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
    - ``--vision``: Whether to use vision-based environment.

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
    :param vision: Whether to use vision-based environment.
        Command line argument has priority over function parameter if both are specified.
    :param show_cfg: Whether to print the configuration.
    :param parser: The argument parser to use. If not specified, a new argument parser will be created.

    :return: MuJoCo Playground environment.

    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments,
        or the task name is invalid.
    :raises ValueError: The number of environments has not been defined, neither by the function parameter nor by the command line arguments.
    :raises ValueError: The episode length has not been defined, neither by the function parameter nor by the command line arguments.
        The task configuration does not provide a default episode length.
    """
    import functools
    from mujoco_playground import registry
    from mujoco_playground._src import wrapper

    import jax

    # parse arguments
    if parser is None:
        parser = argparse.ArgumentParser("MuJoCo Playground")
    parser.add_argument("--task", type=str, default=None, help="Name of the task")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    # playground specific arguments
    parser.add_argument("--episode_length", type=int, default=None, help="Length of the episode")
    parser.add_argument(
        "--action_repeat", type=int, default=None, help="Number of times to repeat the given action per step"
    )
    parser.add_argument(
        "--full_reset",
        action="store_true",
        default=False,
        help="Whether to perform a full reset of the environment on each step, rather than resetting to an initial cached state",
    )
    parser.add_argument("--randomization", action="store_true", default=False, help="Whether to use randomization")
    parser.add_argument("--vision", action="store_true", default=False, help="Whether to use vision-based environment")

    args, _ = parser.parse_known_args()

    task_name = args.task or task_name
    if not task_name:
        raise ValueError(
            "No task name defined. Set the 'task_name' parameter or use '--task <task_name>' as command line argument"
        )
    if task_name not in registry.ALL_ENVS:
        raise ValueError(f"Invalid task name: '{task_name}'. Available tasks: {', '.join(registry.ALL_ENVS)}")
    num_envs = args.num_envs or num_envs
    if num_envs is None:
        raise ValueError(
            "No number of environments defined. Set the 'num_envs' parameter or use '--num_envs <num_envs>' as command line argument"
        )

    # randomization function
    randomization_fn = None
    if args.randomization or randomization:
        randomizer = registry.get_domain_randomizer(task_name)
        if randomizer is not None:
            randomization_fn = functools.partial(randomizer, rng=jax.random.split(config.jax.key, num_envs))

    # adjust environment configuration
    env_cfg = registry.get_default_config(task_name)
    # - episode_length
    episode_length = args.episode_length or episode_length or env_cfg.get("episode_length")
    if episode_length is None:
        raise ValueError(
            f"No episode length defined for the task '{task_name}'. "
            f"Set the 'episode_length' parameter or use '--episode_length <episode_length>' as command line argument"
        )
    # - action_repeat
    action_repeat = args.action_repeat or action_repeat or env_cfg.get("action_repeat", 1)
    # - full_reset
    full_reset = args.full_reset or full_reset
    # - vision
    vision = args.vision or vision

    # load environment
    env = wrapper.wrap_for_brax_training(
        registry.load(task_name, config=env_cfg),
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=randomization_fn,
        full_reset=full_reset,
        vision=vision,
        num_vision_envs=num_envs,
    )

    # set number of environments
    env.num_envs = num_envs
    env.env.num_envs = num_envs
    env.env.unwrapped.num_envs = num_envs

    # print config
    if show_cfg:
        print(f"\nMuJoCo Playground environment")
        print(f"  task: {task_name}")
        print(f"  task config:")
        print(textwrap.indent(str(env_cfg).strip(), prefix=" " * 4))
        print(f"  loading config:")
        print(f"    num_envs: {num_envs}")
        print(f"    episode_length: {episode_length}")
        print(f"    action_repeat: {action_repeat}")
        print(f"    full_reset: {full_reset}")
        print(f"    randomization: {randomization} (function: {randomization_fn})")
        print(f"    vision: {vision}\n")

    set_seed(args.seed)
    return env
