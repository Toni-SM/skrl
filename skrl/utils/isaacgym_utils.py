import os
import sys

from hydra.types import RunMode
from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_automatic_config_search_path, get_args_parser

from omegaconf import DictConfig, OmegaConf

import isaacgym
import isaacgymenvs


def _omegaconf_to_dict(config):
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

def load_isaacgymenv(task_name: str = "", isaacgymenvs_path: str = "", show_cfg: bool = True):
    """
    Loads an Isaac Gym environment (preview 3)

    Parameters
    ----------
        task_name: str, optional
            The name of the task (default: "").
            If not specified, the task name is taken from the command line argument (task=<task_name>).
            Command line argument has priority over function parameter if both are specified
        isaacgymenvs_path: str, optional 
            The path to the isaacgymenvs directory (default: "").
            If empty, the path will obtained from isaacgymenvs package metadata
    Returns
    -------
    isaacgymenvs.tasks.base.vec_task.VecTask
        Isaac Gym environment (preview 3)
    """
    # check task from command line arguments
    defined = False
    for arg in sys.argv:
        if arg.startswith("task="):
            defined = True
            break
    # get task name from command line arguments
    if defined:
        if task_name and task_name != arg.split("=")[1]:
            print("[WARNING] Overriding task name ({}) with command line argument ({})".format(task_name, arg.split("=")[1]))
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
