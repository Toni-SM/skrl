import os
import subprocess
import warnings
import hypothesis
import hypothesis.strategies as st
import pytest


EXAMPLE_DIR = "deepmind"
SCRIPTS = ["dm_suite_cartpole_swingup_ddpg.py",
           "dm_manipulation_stack_sac.py", ""]
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "source", "examples"))
COMMANDS = [f"python {os.path.join(EXAMPLES_DIR, EXAMPLE_DIR, script)}" for script in SCRIPTS]


@pytest.mark.parametrize("command", COMMANDS)
def test_scripts(capsys, command):
    try:
        import gym
    except ImportError as e:
        warnings.warn(f"\n\nUnable to import dm_control environments ({e}).\nThis test will be skipped\n")
        return

    subprocess.run(command, shell=True, check=True)
