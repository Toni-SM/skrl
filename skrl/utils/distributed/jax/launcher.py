from typing import Mapping, Sequence

import argparse
import multiprocessing as mp
import os
import subprocess
import sys


def _get_args_parser() -> argparse.ArgumentParser:
    """Instantiate and configure the argument parser object

    :return: Argument parser object
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description="JAX Distributed Training Launcher")

    # worker/node size related arguments
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--nproc-per-node", "--nproc_per_node", type=int, default=1, help="Number of workers per node")
    parser.add_argument("--node-rank", "--node_rank", type=int, default=0, help="Node rank for multi-node distributed training")

    # coordinator related arguments
    parser.add_argument("--coordinator-address", "--coordinator_address", type=str, default="127.0.0.1:5000",
                        help="IP address and port where process 0 will start a JAX service")

    # positional arguments
    parser.add_argument("script", type=str, help="Training script path to be launched in parallel")
    parser.add_argument("script_args", nargs="...", help="Arguments for the training script")

    return parser

def _start_processes(cmd: Sequence[str], envs: Sequence[Mapping[str, str]], nprocs: int, daemon: bool = False, start_method: str = "spawn") -> None:
    """Start child processes according the specified configuration and wait for them to join

    :param cmd: Command to run on each child process
    :type cmd: list of str
    :param envs: List of environment variables for each child process
    :type envs: list of dictionaries
    :param nprocs: Number of child processes to start
    :type nprocs: int
    :param daemon: Whether the child processes are daemonic (default: ``False``).
                   See Python multiprocessing module for more details
    :type daemon: bool
    :param start_method: Method which should be used to start child processes (default: ``"spawn"``).
                         See Python multiprocessing module for more details
    :type start_method: str
    """
    mp.set_start_method(method=start_method, force=True)

    processes = []
    for i in range(nprocs):
        process = mp.Process(target=_process, args=(cmd, envs[i]), daemon=daemon)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

def _process(cmd: Sequence[str], env: Mapping[str, str]) -> None:
    """Run a command in the current process

    :param cmd: Command to run
    :type cmd: list of str
    :param envs: Environment variables for the current process
    :type envs: dict
    """
    subprocess.run(cmd, env=env)

def launch():
    """Main entry point for launching distributed runs"""
    args = _get_args_parser().parse_args()

    # validate distributed config
    if args.nnodes < 1:
        print(f"[ERROR] Number of nodes ({args.nnodes}) must be greater than 0")
        exit()
    if args.node_rank >= args.nnodes:
        print(f"[ERROR] Node rank ({args.node_rank}) is out of range for the available number of nodes ({args.nnodes})")
        exit()

    # define custom environment variables (see skrl.config.jax for more details)
    envs = []
    for i in range(args.nnodes):
        if i == args.node_rank:
            for j in range(args.nproc_per_node):
                env = os.environ.copy()
                env["JAX_LOCAL_RANK"] = str(j)
                env["JAX_RANK"] = str(i * args.nproc_per_node + j)
                env["JAX_WORLD_SIZE"] = str(args.nnodes * args.nproc_per_node)
                env["JAX_COORDINATOR_ADDR"] = args.coordinator_address.split(":")[0]
                env["JAX_COORDINATOR_PORT"] = args.coordinator_address.split(":")[1]
                envs.append(env)

    # spawn processes
    cmd = [sys.executable, args.script, *args.script_args]
    _start_processes(cmd, envs, nprocs=args.nproc_per_node)
