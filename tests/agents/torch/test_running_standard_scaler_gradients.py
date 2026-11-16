import pytest

import copy

import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("inverse", [False, True])
def test_independent_differentiable_updates(temporal: bool, inverse: bool) -> None:
    """Train an encoder across batches while keeping gradients within each batch."""
    encoder = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        encoder.weight.copy_(torch.tensor([[0.7, -0.2], [0.3, 0.5]]))
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)
    scaler = RunningStandardScaler(size=2, clip_threshold=100.0, device="cpu")
    values = torch.tensor([[-2.0, 3.0], [4.0, 5.0], [1.0, -1.0], [3.0, 2.0]])
    if temporal:
        values = values.reshape(2, 2, 2)

    for step in range(3):
        batch = values + 0.25 * step
        reference_encoder = copy.deepcopy(encoder)
        reference_encoder.zero_grad()
        reference_scaler = RunningStandardScaler(size=2, clip_threshold=100.0, device="cpu")
        reference_scaler.load_state_dict({name: value.clone() for name, value in scaler.state_dict().items()})
        expected = reference_scaler(reference_encoder(batch), train=True, inverse=inverse, no_grad=False)
        expected.square().mean().backward()

        optimizer.zero_grad()
        actual = scaler(encoder(batch), train=True, inverse=inverse, no_grad=False)
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        torch.testing.assert_close(encoder.weight.grad, reference_encoder.weight.grad)
        assert torch.isfinite(encoder.weight.grad).all()
        before = encoder.weight.detach().clone()
        optimizer.step()
        assert not torch.equal(before, encoder.weight.detach())

    for buffer in scaler.buffers():
        assert not buffer.requires_grad
        assert buffer.grad_fn is None


def test_default_gradient_suppression() -> None:
    """Keep the default no-gradient mode unchanged for training and inference."""
    scaler = RunningStandardScaler(size=2, device="cpu")
    values = torch.tensor([[-2.0, 3.0], [4.0, 5.0]], requires_grad=True)
    assert not scaler(values, train=True).requires_grad
    assert not scaler(values).requires_grad
    assert all(not buffer.requires_grad for buffer in scaler.buffers())
