from typing import Optional, Sequence

import os
import sys
from contextlib import contextmanager

from skrl import logger


__all__ = ["load_bidexhands_env"]


@contextmanager
def cwd(new_path: str) -> None:
    """Context manager to change the current working directory

    This function restores the current working directory after the context manager exits

    :param new_path: The new path to change to
    :type new_path: str
    """
    current_path = os.getcwd()
    os.chdir(new_path)
    try:
        yield
    finally:
        os.chdir(current_path)

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


def load_bidexhands_env(task_name: str = "",
                        num_envs: Optional[int] = None,
                        headless: Optional[bool] = None,
                        cli_args: Sequence[str] = [],
                        bidexhands_path: str = "",
                        show_cfg: bool = True):
    """Load a Bi-DexHands environment

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
    :param cli_args: Isaac Gym environment configuration and command line arguments (default: ``[]``)
    :type cli_args: list of str, optional
    :param bidexhands_path: The path to the ``bidexhands`` directory (default: ``""``).
                            If empty, the path will obtained from bidexhands package metadata
    :type bidexhands_path: str, optional
    :param show_cfg: Whether to print the configuration (default: ``True``)
    :type show_cfg: bool, optional

    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments
    :raises RuntimeError: The bidexhands package is not installed or the path is wrong

    :return: Bi-DexHands environment (preview 4)
    :rtype: isaacgymenvs.tasks.base.vec_task.VecTask
    """
    import isaacgym  # isort:skip
    import bidexhands

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
            logger.warning("Overriding num_envs with command line argument --num_envs")
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
            logger.warning("Overriding headless with command line argument --headless")
    # get headless from function arguments
    elif headless is not None:
        sys.argv.append("--headless")

    # others command line arguments
    sys.argv += cli_args

    # get bidexhands path from bidexhands package metadata
    if not bidexhands_path:
        if not hasattr(bidexhands, "__path__"):
            raise RuntimeError("bidexhands package is not installed")
        path = list(bidexhands.__path__)[0]
    else:
        path = bidexhands_path

    sys.path.append(path)

    status = True
    try:
        from utils.config import get_args, load_cfg, parse_sim_params  # type: ignore
        from utils.parse_task import parse_task  # type: ignore
        from utils.process_marl import get_AgentIndex  # type: ignore
    except Exception as e:
        status = False
        logger.error(f"Failed to import required packages: {e}")
    if not status:
        raise RuntimeError(f"The path ({path}) is not valid")

    args = get_args()

    # print config
    if show_cfg:
        print(f"\nBi-DexHands environment ({args.task})")
        _print_cfg(vars(args))

    # update task arguments
    args.task_type = "MultiAgent"  # TODO: get from parameters
    args.cfg_train = os.path.join(path, args.cfg_train)
    args.cfg_env = os.path.join(path, args.cfg_env)

    # load environment
    with cwd(path):
        cfg, cfg_train, _ = load_cfg(args)
        agent_index = get_AgentIndex(cfg)
        sim_params = parse_sim_params(args, cfg, cfg_train)
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    return env
