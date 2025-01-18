import hypothesis
import hypothesis.strategies as st
import pytest
import warnings

import os
import subprocess


# See the following link for Omniverse Isaac Sim Python environment
# https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_python.html
PYTHON_ENVIRONMENT = "./python.sh"

EXAMPLE_DIR = "omniisaacgym"
SCRIPTS = ["ppo_cartpole.py"]
EXAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "source", "examples")
)
COMMANDS = [
    f"{PYTHON_ENVIRONMENT} {os.path.join(EXAMPLES_DIR, EXAMPLE_DIR, script)} headless=True num_envs=64"
    for script in SCRIPTS
]


@pytest.mark.parametrize("command", COMMANDS)
def test_scripts(capsys, command):
    try:
        import omniisaacgymenvs
    except ImportError as e:
        warnings.warn(f"\n\nUnable to import omniisaacgymenvs ({e}).\nThis test will be skipped\n")
        return

    subprocess.run(command, shell=True, check=True)
