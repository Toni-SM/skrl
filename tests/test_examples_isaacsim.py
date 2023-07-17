import os
import subprocess
import warnings
import hypothesis
import hypothesis.strategies as st
import pytest


# See the following link for Omniverse Isaac Sim Python environment
# https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html
PYTHON_ENVIRONMENT = "./python.sh"

EXAMPLE_DIR = "isaacsim"
SCRIPTS = ["cartpole_example_skrl.py"]
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "source", "examples"))
COMMANDS = [f"{PYTHON_ENVIRONMENT} {os.path.join(EXAMPLES_DIR, EXAMPLE_DIR, script)}" for script in SCRIPTS]


@pytest.mark.parametrize("command", COMMANDS)
def test_scripts(capsys, command):
    try:
        from omni.isaac.kit import SimulationApp
    except ImportError as e:
        warnings.warn(f"\n\nUnable to import SimulationApp ({e}).\nThis test will be skipped\n")
        return

    subprocess.run(command, shell=True, check=True)
