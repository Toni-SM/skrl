from typing import Optional, Sequence

import os
import sys

from skrl import logger


__all__ = ["load_isaac_orbit_env"]


def _print_cfg(d, indent=0) -> None:
    """Print the environment configuration

    :param d: The dictionary to print
    :type d: dict
    :param indent: The indentation level (default: ``0``)
    :type indent: int, optional
    """
    for key, value in d.items():
        if isinstance(value, dict):
            _print_cfg(value, indent + 1)
        else:
            print("  |   " * indent + f"  |-- {key}: {value}")


def load_isaac_orbit_env(task_name: str = "",
                         num_envs: Optional[int] = None,
                         headless: Optional[bool] = None,
                         cli_args: Sequence[str] = [],
                         show_cfg: bool = True):
    """Load an Isaac Orbit environment

    Isaac Orbit: https://isaac-orbit.github.io/orbit/index.html

    This function includes the definition and parsing of command line arguments used by Isaac Orbit:

    - ``--headless``: Force display off at all times
    - ``--cpu``: Use CPU pipeline
    - ``--num_envs``: Number of environments to simulate
    - ``--task``: Name of the task
    - ``--num_envs``: Seed used for the environment

    :param task_name: The name of the task (default: ``""``).
                      If not specified, the task name is taken from the command line argument (``--task TASK_NAME``).
                      Command line argument has priority over function parameter if both are specified
    :type task_name: str, optional
    :param num_envs: Number of parallel environments to create (default: ``None``).
                     If not specified, the default number of environments defined in the task configuration is used.
                     Command line argument has priority over function parameter if both are specified
    :type num_envs: int, optional
    :param headless: Whether to use headless mode (no rendering) (default: ``None``).
                     If not specified, the default task configuration is used.
                     Command line argument has priority over function parameter if both are specified
    :type headless: bool, optional
    :param cli_args: Isaac Orbit configuration and command line arguments (default: ``[]``)
    :type cli_args: list of str, optional
    :param show_cfg: Whether to print the configuration (default: ``True``)
    :type show_cfg: bool, optional

    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments

    :return: Isaac Orbit environment
    :rtype: gym.Env
    """
    import argparse
    import atexit
    import gym

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
            raise ValueError("No task name defined. Set the task_name parameter or use --task <task_name> as command line argument")
        if task_name and task_name != sys.argv[arg_index]:
            logger.warning(f"Overriding task ({task_name}) with command line argument ({sys.argv[arg_index]})")
    # get task name from function arguments
    else:
        if task_name:
            sys.argv.append("--task")
            sys.argv.append(task_name)
        else:
            raise ValueError("No task name defined. Set the task_name parameter or use --task <task_name> as command line argument")

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

    # check headless from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--headless"):
            defined = True
            break
    # get headless from command line arguments
    if defined:
        if headless is not None:
            logger.warning("Overriding headless with command line argument (--headless)")
    # get headless from function arguments
    elif headless is not None:
        sys.argv.append("--headless")

    # others command line arguments
    sys.argv += cli_args

    # parse arguments
    parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
    parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
    parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--task", type=str, default=None, help="Name of the task.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    args = parser.parse_args()

    # load the most efficient kit configuration in headless mode
    if args.headless:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
    else:
        app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"

    # launch the simulator
    from omni.isaac.kit import SimulationApp  # type: ignore

    config = {"headless": args.headless}
    simulation_app = SimulationApp(config, experience=app_experience)

    @atexit.register
    def close_the_simulator():
        simulation_app.close()

    # import orbit extensions
    import omni.isaac.contrib_envs  # type: ignore
    import omni.isaac.orbit_envs  # type: ignore
    from omni.isaac.orbit_envs.utils import parse_env_cfg  # type: ignore

    cfg = parse_env_cfg(args.task, use_gpu=not args.cpu, num_envs=args.num_envs)

    # print config
    if show_cfg:
        print(f"\nIsaac Orbit environment ({args.task})")
        try:
            _print_cfg(cfg)
        except AttributeError as e:
            pass

    # load environment
    env = gym.make(args.task, cfg=cfg, headless=args.headless)

    return env
