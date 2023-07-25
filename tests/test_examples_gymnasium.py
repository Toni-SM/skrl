import os
import subprocess
import warnings
import hypothesis
import hypothesis.strategies as st
import pytest


EXAMPLE_DIR = "gymnasium"
SCRIPTS = ["ddpg_gymnasium_pendulum.py",
           "cem_gymnasium_cartpole.py",
           "dqn_gymnasium_cartpole.py",
           "q_learning_gymnasium_frozen_lake.py"]
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "source", "examples"))
COMMANDS = [f"python {os.path.join(EXAMPLES_DIR, EXAMPLE_DIR, script)}" for script in SCRIPTS]


@pytest.mark.parametrize("command", COMMANDS)
def test_scripts(capsys, command):
    try:
        import gymnasium
    except ImportError as e:
        warnings.warn(f"\n\nUnable to import gymnasium ({e}).\nThis test will be skipped\n")
        return

    subprocess.run(command, shell=True, check=True)
