import os
import subprocess
import warnings
import hypothesis
import hypothesis.strategies as st
import pytest


EXAMPLE_DIR = "shimmy"
SCRIPTS = ["dqn_shimmy_atari_pong.py",
           "sac_shimmy_dm_control_acrobot_swingup_sparse.py",
           "ddpg_openai_gym_compatibility_pendulum.py"]
EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "source", "examples"))
COMMANDS = [f"python {os.path.join(EXAMPLES_DIR, EXAMPLE_DIR, script)}" for script in SCRIPTS]


@pytest.mark.parametrize("command", COMMANDS)
def test_scripts(capsys, command):
    try:
        import shimmy
    except ImportError as e:
        warnings.warn(f"\n\nUnable to import shimmy ({e}).\nThis test will be skipped\n")
        return

    subprocess.run(command, shell=True, check=True)
