from typing import Any

import gymnasium


def _sample_space(space: gymnasium.Space, batch_size: int) -> Any:
    """Sample a batch while advancing the original space's random generator."""
    batched_space = gymnasium.vector.utils.batch_space(space, batch_size)
    sample = batched_space.sample()
    # batch_space copies the generator, so retain its progress for the next call.
    space.np_random.bit_generator.state = batched_space.np_random.bit_generator.state
    return sample
