import os
import sys

__all__ = ["load_isaac_orbit_env"]


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


def load_isaac_orbit_env(task_name: str = "", show_cfg: bool = True):
    """Load an Isaac Orbit environment

    Isaac Orbit: https://isaac-orbit.github.io/orbit/index.html

    This function includes the definition and parsing of command line arguments used by Isaac Orbit:

    - ``--headless``: Force display off at all times
    - ``--cpu``: Use CPU pipeline
    - ``--num_envs``: Number of environments to simulate
    - ``--task``: Name of the task
    - ``--num_envs``: Seed used for the environment

    :param task_name: The name of the task (default: "").
                      If not specified, the task name is taken from the command line argument (``--task TASK_NAME``).
                      Command line argument has priority over function parameter if both are specified
    :type task_name: str, optional
    :param show_cfg: Whether to print the configuration (default: True)
    :type show_cfg: bool, optional

    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments

    :return: Isaac Orbit environment
    :rtype: gym.Env
    """
    import gym
    import atexit
    import argparse

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
            print("[WARNING] Overriding task ({}) with command line argument ({})".format(task_name, sys.argv[arg_index]))
    # get task name from function arguments
    else:
        if task_name:
            sys.argv.append("--task")
            sys.argv.append(task_name)
        else:
            raise ValueError("No task name defined. Set the task_name parameter or use --task <task_name> as command line argument")

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
    from omni.isaac.kit import SimulationApp

    config = {"headless": args.headless}
    simulation_app = SimulationApp(config, experience=app_experience)

    @atexit.register
    def close_the_simulator():
        simulation_app.close()

    # import orbit extensions
    import omni.isaac.contrib_envs  # noqa: F401
    import omni.isaac.orbit_envs  # noqa: F401
    from omni.isaac.orbit_envs.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, use_gpu=not args.cpu, num_envs=args.num_envs)

    # print config
    if show_cfg:
        print("\nIsaac Orbit environment ({})".format(args.task))
        try:
            _print_cfg(cfg)
        except AttributeError as e:
            pass

    # load environment
    env = gym.make(args.task, cfg=cfg, headless=args.headless)

    return env
