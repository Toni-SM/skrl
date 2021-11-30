skrl: Reinforcement Learning at your service
============================================

**skrl** is a Python library for reinforcement learning lovers 

.. note::

  This project is under active development

Contents
--------

.. toctree::
  :maxdepth: 1
  :caption: Agents

  modules/skrl.agents.base_class
  modules/skrl.agents.ppo
  modules/skrl.agents.ddpg
  modules/skrl.agents.td3
  modules/skrl.agents.sac

.. toctree::
  :maxdepth: 1
  :caption: Environments

  modules/skrl.envs.wrapping
  modules/skrl.envs.isaac_gym

.. toctree::
  :maxdepth: 1
  :caption: Memories

  modules/skrl.memories.base_class
  modules/skrl.memories.random
  modules/skrl.memories.prioritized

.. toctree::
  :maxdepth: 1
  :caption: Models

  modules/skrl.models.base_class
  modules/skrl.models.categorical
  modules/skrl.models.gaussian 
  modules/skrl.models.deterministic 

.. toctree::
  :maxdepth: 1
  :caption: Noises
      
  modules/skrl.noises.base_class
  modules/skrl.noises.gaussian
  modules/skrl.noises.ornstein_uhlenbeck

.. toctree::
  :maxdepth: 1
  :caption: Trainers
      
  modules/skrl.trainers.base_class
  modules/skrl.trainers.sequential
  modules/skrl.trainers.concurrent
