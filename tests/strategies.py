import hypothesis.strategies as st

import gymnasium


@st.composite
def gymnasium_space_stategy(draw, space_type: str = "", remaining_iterations: int = 5) -> gymnasium.spaces.Space:
    if not space_type:
        space_type = draw(st.sampled_from(["Box", "Discrete", "MultiDiscrete", "Dict", "Tuple"]))
    # recursion base case
    if remaining_iterations <= 0 and space_type in ["Dict", "Tuple"]:
        space_type = "Box"

    if space_type == "Box":
        shape = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=5))
        return gymnasium.spaces.Box(low=-1, high=1, shape=shape)
    elif space_type == "Discrete":
        n = draw(st.integers(min_value=1, max_value=5))
        return gymnasium.spaces.Discrete(n)
    elif space_type == "MultiDiscrete":
        nvec = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=5))
        return gymnasium.spaces.MultiDiscrete(nvec)
    elif space_type == "Dict":
        remaining_iterations -= 1
        keys = draw(st.lists(st.text(st.characters(codec="ascii"), min_size=1, max_size=5), min_size=1, max_size=3))
        spaces = {key: draw(gymnasium_space_stategy(remaining_iterations=remaining_iterations)) for key in keys}
        return gymnasium.spaces.Dict(spaces)
    elif space_type == "Tuple":
        remaining_iterations -= 1
        spaces = draw(
            st.lists(gymnasium_space_stategy(remaining_iterations=remaining_iterations), min_size=1, max_size=3)
        )
        return gymnasium.spaces.Tuple(spaces)
    else:
        raise ValueError(f"Invalid space type: {space_type}")


@st.composite
def gym_space_stategy(draw, space_type: str = "", remaining_iterations: int = 5) -> "gym.spaces.Space":
    import gym

    if not space_type:
        space_type = draw(st.sampled_from(["Box", "Discrete", "MultiDiscrete", "Dict", "Tuple"]))
    # recursion base case
    if remaining_iterations <= 0 and space_type in ["Dict", "Tuple"]:
        space_type = "Box"

    if space_type == "Box":
        shape = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=5))
        return gym.spaces.Box(low=-1, high=1, shape=shape)
    elif space_type == "Discrete":
        n = draw(st.integers(min_value=1, max_value=5))
        return gym.spaces.Discrete(n)
    elif space_type == "MultiDiscrete":
        nvec = draw(st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=5))
        return gym.spaces.MultiDiscrete(nvec)
    elif space_type == "Dict":
        remaining_iterations -= 1
        keys = draw(st.lists(st.text(st.characters(codec="ascii"), min_size=1, max_size=5), min_size=1, max_size=3))
        spaces = {key: draw(gym_space_stategy(remaining_iterations=remaining_iterations)) for key in keys}
        return gym.spaces.Dict(spaces)
    elif space_type == "Tuple":
        remaining_iterations -= 1
        spaces = draw(st.lists(gym_space_stategy(remaining_iterations=remaining_iterations), min_size=1, max_size=3))
        return gym.spaces.Tuple(spaces)
    else:
        raise ValueError(f"Invalid space type: {space_type}")
