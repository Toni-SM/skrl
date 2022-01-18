import os
import sys
from contextlib import contextmanager

__all__ = ["load_isaacgym_env_preview2", "load_isaacgym_env_preview3"]


@contextmanager
def cwd(new_path):
    current_path = os.getcwd()
    os.chdir(new_path)
    try:
        yield
    finally:
        os.chdir(current_path)

def _omegaconf_to_dict(config):
    from omegaconf import DictConfig

    d = {}
    for k, v in config.items():
        d[k] = _omegaconf_to_dict(v) if isinstance(v, DictConfig) else v
    return d

def _print_cfg(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            _print_cfg(value, indent + 1)
        else:
            print('  |   ' * indent + "  |-- {}: {}".format(key, value))


def load_isaacgym_env_preview2(task_name: str = "", isaacgymenvs_path: str = "", show_cfg: bool = True):
    """Loads an Isaac Gym environment (preview 2)

    :param task_name: The name of the task (default: "").
                      If not specified, the task name is taken from the command line argument (``--task TASK_NAME``).
                      Command line argument has priority over function parameter if both are specified
    :type task_name: str, optional
    :param isaacgymenvs_path: The path to the ``rlgpu`` directory (default: "").
                              If empty, the path will obtained from isaacgym package metadata
    :type isaacgymenvs_path: str, optional
    :param show_cfg: Whether to print the configuration (default: True)
    :type show_cfg: bool, optional
    
    :raises ValueError: The task name has not been defined, 
                        neither by the function parameter nor by the command line arguments
    :raises RuntimeError: The isaacgym package is not installed or the path is wrong

    :return: Isaac Gym environment (preview 2)
    :rtype: tasks.base.vec_task.VecTask
    """
    import isaacgym

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
    
    # get isaacgym envs path from isaacgym package metadata
    if not isaacgymenvs_path:
        if not hasattr(isaacgym, "__path__"):
            raise RuntimeError("isaacgym package is not installed")
        path = isaacgym.__path__
        path = os.path.join(path[0], "..", "rlgpu")
    else:
        path = isaacgymenvs_path

    # import required packages
    sys.path.append(path)

    status = True
    try:
        from utils.config import get_args, load_cfg, parse_sim_params
        from utils.parse_task import parse_task
    except Exception as e:
        status = False
        print("[ERROR] Failed to import required packages: {}".format(e))
    if not status:
        raise RuntimeError("The path ({}) is not valid or the isaacgym package is not installed in editable mode (pip install -e .)" \
            .format(path))

    args = get_args()

    # print config
    if show_cfg:
        print("\nIsaac Gym environment ({})".format(args.task))
        _print_cfg(vars(args))
   
    # update task arguments
    args.cfg_train = os.path.join(path, args.cfg_train)
    args.cfg_env = os.path.join(path, args.cfg_env)

    # load environment
    with cwd(path):
        cfg, cfg_train, _ = load_cfg(args)
        sim_params = parse_sim_params(args, cfg, cfg_train)
        task, env = parse_task(args, cfg, cfg_train, sim_params)
    
    return env

def load_isaacgym_env_preview3(task_name: str = "", isaacgymenvs_path: str = "", show_cfg: bool = True):
    """Loads an Isaac Gym environment (preview 3) 
    
    Isaac Gym benchmark environments: https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

    :param task_name: The name of the task (default: "").
                      If not specified, the task name is taken from the command line argument (``task=TASK_NAME``).
                      Command line argument has priority over function parameter if both are specified
    :type task_name: str, optional
    :param isaacgymenvs_path: The path to the ``isaacgymenvs`` directory (default: "").
                              If empty, the path will obtained from isaacgymenvs package metadata
    :type isaacgymenvs_path: str, optional
    :param show_cfg: Whether to print the configuration (default: True)
    :type show_cfg: bool, optional
    
    :raises ValueError: The task name has not been defined, neither by the function parameter nor by the command line arguments
    :raises RuntimeError: The isaacgymenvs package is not installed or the path is wrong

    :return: Isaac Gym environment (preview 3)
    :rtype: isaacgymenvs.tasks.base.vec_task.VecTask
    """
    from hydra.types import RunMode
    from hydra._internal.hydra import Hydra
    from hydra._internal.utils import create_automatic_config_search_path, get_args_parser

    from omegaconf import OmegaConf

    import isaacgym
    import isaacgymenvs
    
    # check task from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("task="):
            defined = True
            break
    # get task name from command line arguments
    if defined:
        if task_name and task_name != arg.split("task=")[1].split(" ")[0]:
            print("[WARNING] Overriding task name ({}) with command line argument ({})" \
                .format(task_name, arg.split("task=")[1].split(" ")[0]))
    # get task name from function arguments
    else:
        if task_name:
            sys.argv.append("task={}".format(task_name))
        else:
            raise ValueError("No task name defined. Set task_name parameter or use task=<task_name> as command line argument")

    # get isaacgymenvs path from isaacgymenvs package metadata
    if isaacgymenvs_path == "":
        if not hasattr(isaacgymenvs, "__path__"):
            raise RuntimeError("isaacgymenvs package is not installed")
        isaacgymenvs_path = list(isaacgymenvs.__path__)[0]
    config_path = os.path.join(isaacgymenvs_path, "cfg")

    # set omegaconf resolvers
    OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver('if', lambda condition, a, b: a if condition else b)
    OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)

    # get hydra config without use @hydra.main
    config_file = "config"
    args = get_args_parser().parse_args()
    search_path = create_automatic_config_search_path(config_file, None, config_path)
    hydra_object = Hydra.create_main_hydra2(task_name='load_isaacgymenv', config_search_path=search_path)
    config = hydra_object.compose_config(config_file, args.overrides, run_mode=RunMode.RUN)

    cfg = _omegaconf_to_dict(config.task)

    # print config
    if show_cfg:
        print("\nIsaac Gym environment ({})".format(config.task.name))
        _print_cfg(cfg)

    # load environment
    sys.path.append(isaacgymenvs_path)
    from tasks import isaacgym_task_map
    env = isaacgym_task_map[config.task.name](cfg=cfg, 
                                              sim_device=config.sim_device,
                                              graphics_device_id=config.graphics_device_id,
                                              headless=config.headless)
    return env
