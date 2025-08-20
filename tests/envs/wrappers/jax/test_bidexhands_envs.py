import pytest

from skrl.envs.wrappers.jax import BiDexHandsWrapper, wrap_env


def test_env(capsys: pytest.CaptureFixture):
    # check wrapper definition
    assert isinstance(wrap_env(None, "bidexhands"), BiDexHandsWrapper)
