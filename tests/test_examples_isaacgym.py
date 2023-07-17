import os
import subprocess
import warnings
import hypothesis
import hypothesis.strategies as st
import pytest


EXAMPLE_DIR = "isaacgym"
SCRIPTS = ["ppo_cartpole.py",
           "trpo_cartpole.py"]
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "source", "examples"))
COMMANDS = [f"python {os.path.join(EXAMPLES_DIR, EXAMPLE_DIR, script)} headless=True num_envs=64" for script in SCRIPTS]


@pytest.mark.parametrize("command", COMMANDS)
def test_scripts(capsys, command):
    try:
        import isaacgymenvs
    except ImportError as e:
        warnings.warn(f"\n\nUnable to import isaacgymenvs ({e}).\nThis test will be skipped\n")
        return

    subprocess.run(command, shell=True, check=True)
