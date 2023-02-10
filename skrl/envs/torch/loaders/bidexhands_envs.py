import os
import sys
import subprocess
from contextlib import contextmanager

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
    :param indent: The indentation level (default: 0)
    :type indent: int, optional
    """
    for key, value in d.items():
        if isinstance(value, dict):
            _print_cfg(value, indent + 1)
        else:
            print('  |   ' * indent + "  |-- {}: {}".format(key, value))

def _get_package_location(package_name):
    try:
        output = subprocess.check_output(["pip", "show", package_name])
        for line in output.decode().split("\n"):
            if line.startswith("Location:"):
                return line.split(":")[1].strip()
    except Exception as e:
        print(e)
    return ""

def load_bidexhands_env(task_name: str = "", bidexhands_path: str = "", show_cfg: bool = True):
    """Load a Bi-DexHands environment

    :param task_name: The name of the task (default: "").
                      If not specified, the task name is taken from the command line argument (``--task=TASK_NAME``).
                      Command line argument has priority over function parameter if both are specified
    :type task_name: str, optional
    :param bidexhands_path: The path to the ``bi-dexhands`` directory (default: "").
                            If empty, the path will obtained from bidexhands package metadata
    :type bidexhands_path: str, optional
    :param show_cfg: Whether to print the configuration (default: True)
    :type show_cfg: bool, optional

    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments
    :raises RuntimeError: The bidexhands package is not installed or the path is wrong

    :return: Bi-DexHands environment (preview 3)
    :rtype: isaacgymenvs.tasks.base.vec_task.VecTask
    """
    import isaacgym

    # check task from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("--task"):
            defined = True
            break
    # get task name from function arguments
    if not defined:
        if task_name:
            sys.argv.append("--task={}".format(task_name))
        else:
            raise ValueError("No task name defined. Set the task_name parameter or use --task=<task_name> as command line argument")

    # get bidexhands path from bidexhands package metadata
    path = bidexhands_path if bidexhands_path else os.path.join(_get_package_location("bidexhands"), "bi-dexhands")
    if path:
        sys.path.append(path)

    status = True
    try:
        from utils.config import get_args, load_cfg, parse_sim_params  # type: ignore
        from utils.parse_task import parse_task   # type: ignore
        from utils.process_marl import get_AgentIndex  # type: ignore
    except Exception as e:
        status = False
        print("[ERROR] Failed to import required packages: {}".format(e))
    if not status:
        raise RuntimeError("The path ({}) is not valid" \
            .format(path))

    args = get_args()

    # print config
    if show_cfg:
        print("\nBi-DexHands environment ({})".format(args.task))
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
