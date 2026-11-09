from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys


def _get_args_parser() -> argparse.ArgumentParser:
    """Instantiate and configure the argument parser object.

    :return: Argument parser object.
    """
    parser = argparse.ArgumentParser(description="JAX Distributed Training Launcher")

    # worker/node size related arguments
    parser.add_argument("--nnodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--nproc-per-node", "--nproc_per_node", type=int, default=1, help="Number of workers per node")
    parser.add_argument(
        "--node-rank", "--node_rank", type=int, default=0, help="Node rank for multi-node distributed training"
    )

    # coordinator related arguments
    parser.add_argument(
        "--coordinator-address",
        "--coordinator_address",
        type=str,
        default="127.0.0.1:5000",
        help="IP address and port where process 0 will start a JAX service",
    )

    # positional arguments
    parser.add_argument("script", type=str, help="Training script path to be launched in parallel")
    parser.add_argument("script_args", nargs="...", help="Arguments for the training script")

    return parser


def _start_processes(
    cmd: list[str],
    envs: list[dict[str, str]],
    nprocs: int,
    daemon: bool = False,
    start_method: str = "spawn",
) -> None:
    """Start child processes according the specified configuration and wait for them to join.

    :param cmd: Command to run on each child process.
    :param envs: List of environment variables for each child process.
    :param nprocs: Number of child processes to start.
    :param daemon: Whether the child processes are daemonic. See Python multiprocessing module for more details.
    :param start_method: Method used to start child processes. See Python multiprocessing module for more details.
    """
    context = mp.get_context(start_method)

    processes = []
    for i in range(nprocs):
        process = context.Process(target=_process, args=(cmd, envs[i]), daemon=daemon)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    for process in processes:
        if process.exitcode:
            raise SystemExit(process.exitcode if process.exitcode > 0 else 128 - process.exitcode)


def _process(cmd: list[str], env: dict[str, str]) -> None:
    """Run a command in the current process.

    :param cmd: Command to run.
    :param env: Environment variables for the current process.
    """
    result = subprocess.run(cmd, env=env)
    if result.returncode:
        raise SystemExit(result.returncode if result.returncode > 0 else 128 - result.returncode)


def launch():
    """Main entry point for launching distributed runs."""
    parser = _get_args_parser()
    args = parser.parse_args()

    # validate distributed config
    if args.nnodes < 1 or args.nproc_per_node < 1:
        parser.error("nnodes and nproc-per-node must both be positive")
    if not 0 <= args.node_rank < args.nnodes:
        parser.error("node-rank must be between zero and nnodes - 1")
    address, _, port = args.coordinator_address.rpartition(":")
    if not address or not port.isascii() or not port.isdecimal() or not 0 < int(port) <= 65535:
        parser.error("coordinator-address must include a host and a port in the range 1..65535")

    # define custom environment variables (see skrl.config.jax for more details)
    envs = []
    for i in range(args.nnodes):
        if i == args.node_rank:
            for j in range(args.nproc_per_node):
                env = os.environ.copy()
                env["JAX_LOCAL_RANK"] = str(j)
                env["JAX_RANK"] = str(i * args.nproc_per_node + j)
                env["JAX_WORLD_SIZE"] = str(args.nnodes * args.nproc_per_node)
                env["JAX_COORDINATOR_ADDR"] = address
                env["JAX_COORDINATOR_PORT"] = port
                envs.append(env)

    # spawn processes
    cmd = [sys.executable, args.script, *args.script_args]
    _start_processes(cmd, envs, nprocs=args.nproc_per_node)
