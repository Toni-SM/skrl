import pytest

import os
import sys

from skrl.utils.distributed.jax import launcher


@pytest.mark.parametrize(
    "option, value",
    [
        ("--nnodes", "0"),
        ("--nproc-per-node", "0"),
        ("--node-rank", "-1"),
        ("--node-rank", "1"),
        ("--coordinator-address", "host"),
        ("--coordinator-address", ":1234"),
        ("--coordinator-address", "host:0"),
        ("--coordinator-address", "host:70000"),
        ("--coordinator-address", "host:abc"),
        ("--coordinator-address", "host:\u0663\u0664"),
        ("--coordinator-address", "[::1]"),
    ],
)
def test_invalid_arguments(monkeypatch, option, value):
    monkeypatch.setattr(sys, "argv", ["launcher", option, value, "train.py"])
    with pytest.raises(SystemExit) as exc_info:
        launcher.launch()
    assert exc_info.value.code == 2


def test_ipv6_coordinator_address(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["launcher", "--coordinator-address", "[::1]:1234", "train.py"])
    monkeypatch.setattr(launcher, "_start_processes", lambda cmd, envs, nprocs: calls.append(envs))
    launcher.launch()
    env = calls[0][0]
    assert (env["JAX_COORDINATOR_ADDR"], env["JAX_COORDINATOR_PORT"]) == ("[::1]", "1234")


def test_worker_success():
    launcher._start_processes([sys.executable, "-c", "pass"], [dict(os.environ)], nprocs=1)


@pytest.mark.parametrize(
    "code, expected",
    [
        ("raise SystemExit(7)", 7),
        pytest.param(
            "import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
            128 + 9,
            marks=pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is not available on Windows"),
        ),
        pytest.param(
            "import os, signal; os.kill(os.getppid(), signal.SIGKILL)",  # kill the multiprocessing worker itself
            128 + 9,
            marks=pytest.mark.skipif(sys.platform == "win32", reason="SIGKILL is not available on Windows"),
        ),
    ],
    ids=["exit-code", "command-killed", "worker-killed"],
)
def test_worker_failure(code, expected):
    with pytest.raises(SystemExit) as exc_info:
        launcher._start_processes([sys.executable, "-c", code], [dict(os.environ)], nprocs=1)
    assert exc_info.value.code == expected
