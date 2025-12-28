import argparse

from skrl import config


def load_playground_env(
    *,
    task_name: str = "",
    num_envs: int | None = None,
    headless: bool | None = None,
    cli_args: list[str] = [],
    show_cfg: bool = True,
    parser: argparse.ArgumentParser | None = None,
):
    """Load a MuJoCo Playground environment.

    MuJoCo Playground: https://playground.mujoco.org
    """
    import functools
    from mujoco_playground import registry
    from mujoco_playground._src import wrapper

    import jax

    # randomization function
    randomizer = registry.get_domain_randomizer(task_name)
    if randomizer is not None:
        randomization_fn = functools.partial(randomizer, rng=jax.random.split(config.jax.key, num_envs))
    else:
        randomization_fn = None

    # load environment
    env_cfg = registry.get_default_config(task_name)
    env = wrapper.wrap_for_brax_training(
        registry.load(task_name, config=env_cfg),
        vision=False,
        num_vision_envs=1,
        episode_length=env_cfg.episode_length,
        action_repeat=1,
        randomization_fn=randomization_fn,
        full_reset=False,
    )

    # set number of environments
    env.num_envs = num_envs
    env.env.num_envs = num_envs
    env.env.unwrapped.num_envs = num_envs

    # print config
    if show_cfg:
        print(f"\nMuJoCo Playground environment ({task_name})")
        print(env_cfg)

    return env
