from datetime import datetime
from typing import  Sequence

import gymnasium
import torch
import gym_envs

from skrl.agents.torch.crossq import CrossQ as Agent
from skrl.agents.torch.crossq import CROSSQ_DEFAULT_CONFIG as DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch.sequential import SequentialTrainer

from models import *


def test_agent():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Joint_PandaReach-v0")
    parser.add_argument("--seed", type=int, default=9572)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--n-steps", type=int, default=30_000)
    
    args = parser.parse_args()
    # env
    env = gymnasium.make(args.env, max_episode_steps=300, render_mode=None)
    env.reset(seed=args.seed)
    set_seed(args.seed, deterministic=True)
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
    print(models)
    # for model in models.values():
    #     model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
        
    # memory
    memory = RandomMemory(memory_size=1_000_000, num_envs=env.num_envs, device=env.device)

    # agent
    cfg = DEFAULT_CONFIG.copy()
    cfg["mixed_precision"] = False
    cfg["experiment"]["wandb"] = args.wandb
    cfg["experiment"]["wandb_kwargs"] = dict(
        name=f"test-crossq-torch-{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}",
        project="skrl",
        entity=None,
        tags="",
        # config=cfg,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    agent = Agent(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )
    # trainer
    cfg_trainer = {
        "timesteps": args.n_steps,
        "headless": True,
        "disable_progressbar": False,
        "close_environment_at_exit": False,
    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    trainer.train()


test_agent()
