from skrl.agents.torch.base import Agent
from skrl.envs.wrappers.torch.base import Wrapper

from typing import Tuple

import numpy as np 
import torch 



def evaluate_single_agent(agent:Agent, env:Wrapper, num_episodes:int=100) -> Tuple[float, float]:
    """
    A stable-baseline3-style evaluation function for evaluating episodic performance of 
    an agent in single agent environment
    
    :param agent: agent to be evaluated
    :type agent: skrl.agents.torch.base.Agent
    :param env: Environment Wrapper
    :type env: skrl.envs.wrappers.torch.base.Wrapper
    :param num_episodes: number of episodes used for evaluation
    :type num_episodes: int
    
    """
    assert env.num_envs == 1, "Only support single environment."
    rewards = []
    for _ in range(num_episodes):
      obs_t, _ = env.reset()
      episode_reward = 0.
      done = False 
      random_timesteps = agent._random_timesteps
      while not done:
        with torch.no_grad():
          agent.pre_interaction(random_timesteps+1, random_timesteps+2)
          agent.set_mode("eval")
          action_t,_,_ = agent.act(obs_t, random_timesteps+1, random_timesteps+2)
        obs_tplus1, reward, terminated, truncated, info = env.step(action_t)
        obs_t = obs_tplus1 
        episode_reward += reward 
        done = terminated or truncated
      rewards.append(episode_reward)
    return np.mean(rewards), np.std(rewards)