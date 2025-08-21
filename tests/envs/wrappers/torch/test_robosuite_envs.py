import pytest

from skrl.envs.wrappers.torch import RobosuiteWrapper, wrap_env


def test_env(capsys: pytest.CaptureFixture):
    # check wrapper definition
    with pytest.raises(AttributeError):
        assert isinstance(wrap_env(None, "robosuite"), RobosuiteWrapper)
