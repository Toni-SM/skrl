from typing import Optional

import argparse
import sys
import time
import gymnasium
import tqdm.rich as tqdm

import numpy as np
import torch

from skrl.agents.torch.crossq import CROSSQ_DEFAULT_CONFIG as DEFAULT_CONFIG
from skrl.agents.torch.crossq import CrossQ as Agent
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.utils import set_seed
from tests.torch.test_crossq_models import *


def test_agent():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="Joint_PandaReach-v0")
    parser.add_argument("--n-timesteps", default=1000)
    parser.add_argument("--steps-per-episode", type=int, default=100)
    parser.add_argument("--log-interval", default=10)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--seed", default=9572)
    parser.add_argument("--verbose", default=1)
    parser.add_argument(
        "--goal-space-size",
        default=2,
        choices=[0, 1, 2],
        help="Goal space size (0 : SMALL box, 1 : MEDIUM box, 2 : LARGE box)",
    )

    args = parser.parse_args()
    set_seed(args.seed)
    # env
    env = gymnasium.make(
        args.env_id,
        goal_space_size=args.goal_space_size,
        max_episode_steps=args.steps_per_episode,
        render_mode="human" if args.gui else None,
    )
    env = wrap_env(env, wrapper="gymnasium")

    models = {}
    models["policy"] = StochasticActor(
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_arch=[256, 256],
        device=env.device,
        use_batch_norm=True,
    )
    models["critic_1"] = Critic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_arch=[1024, 1024],
        device=env.device,
        use_batch_norm=True,
    )
    models["critic_2"] = Critic(
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_arch=[1024, 1024],
        device=env.device,
        use_batch_norm=True,
    )

    # memory
    memory = RandomMemory(memory_size=1, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = DEFAULT_CONFIG

    agent = Agent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )

    # Change the path to the best_agent.pt file you want to load
    agent.load(
        "/home/sora/travail/rhoban/skrl/tests/torch/runs/25-02-24_13-12-11-279869_CrossQ/checkpoints/best_agent.pt"
    )

    # reset env
    states, infos = env.reset()

    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    successes: list[bool] = []
    ep_len = 0

    for timestep in tqdm.tqdm(range(0, args.n_timesteps), file=sys.stdout):
        # pre-interaction
        agent.pre_interaction(timestep=timestep, timesteps=args.n_timesteps)

        with torch.no_grad():
            # compute actions
            outputs = agent.act(states, timestep=timestep, timesteps=args.n_timesteps)
            actions = outputs[0]

            # step the environments
            next_states, rewards, terminated, truncated, infos = env.step(actions)

            # render scene
            if not not args.gui:
                env.render()

        ep_len += 1
        episode_reward += rewards.item()
        done = terminated.any()
        trunc = truncated.any()

        if done or trunc:
            success: Optional[bool] = infos.get("is_success")
            if args.verbose > 0:
                print(f"Infos : {infos}")
                print(f"Episode Reward: {episode_reward:.2f}")
                print("Episode Length", ep_len)
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)

            if success is not None:
                successes.append(success)

            with torch.no_grad():
                states, infos = env.reset()
                episode_reward = 0.0
                ep_len = 0

            continue

        states = next_states

        if args.gui:
            time.sleep(1 / 240)

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(f"{len(episode_rewards)} Episodes")
        print(f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")


test_agent()
