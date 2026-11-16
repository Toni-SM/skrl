import pytest

from gymnasium import spaces

import numpy as np
import warp as wp

from skrl.utils.model_instantiators.warp import gaussian_model


@pytest.mark.parametrize("initial_contents", [0.0, -17.0, float("nan")])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_gaussian_entropy_ignores_uninitialized_buffer_contents(monkeypatch, initial_contents, batch_size):
    model = gaussian_model(
        observation_space=spaces.Box(low=-1, high=1, shape=(2,)),
        action_space=spaces.Box(low=-1, high=1, shape=(3,)),
        device="cpu",
        initial_log_std=0.35,
        network=[{"name": "net", "input": "OBSERVATIONS", "layers": [4], "activations": "tanh"}],
        output="ACTIONS",
    )
    model.init_state_dict(role="policy")
    observations = wp.array(np.zeros((batch_size, 2), dtype=np.float32), device="cpu")
    allocate_empty = wp.empty

    def poisoned_empty(*args, **kwargs):
        array = allocate_empty(*args, **kwargs)
        array.fill_(initial_contents)
        return array

    with wp.Tape() as tape:
        _, outputs = model.act({"observations": observations}, role="policy")
        # Make unspecified allocation contents deterministic without replacing the entropy computation.
        with monkeypatch.context() as context:
            context.setattr(wp, "empty", poisoned_empty)
            entropy = model.get_entropy(outputs["stddev"], role="policy")
    expected = np.sum(0.5 + 0.5 * np.log(2 * np.pi) + np.log(outputs["stddev"].numpy()))
    np.testing.assert_allclose(entropy.numpy(), [expected], rtol=1e-6, atol=1e-6)
    tape.backward(entropy)
    np.testing.assert_allclose(outputs["log_std"].grad.numpy(), np.ones(3), rtol=1e-6, atol=1e-6)
