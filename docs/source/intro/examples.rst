Examples
========

.. contents:: Table of Contents
   :depth: 1
   :local:
   :backlinks: none

Learning in a Gym environment (one agent, one environment)
----------------------------------------------------------

This example performs the training of an agent in OpenAI Gym's `inverted pendulum <https://gym.openai.com/envs/Pendulum-v0/>`_ (*Pendulum-v0*) environment, a classic problem in the continuous domain control literature

.. literalinclude:: ../examples/gym_pendulum.py
    :language: python
    :linenos:
    :emphasize-lines: 13, 49-50

Learning in a Isaac Gym environment (one agent, multiple environments)
----------------------------------------------------------------------

This example performs the training of an agent in Isaac Gym's Cartpole environment. It tries to load the environment from preview 3, but if it fails, it will try to load the environment from preview 2

.. literalinclude:: ../examples/isaacgym_cartpole.py
    :language: python
    :linenos:
    :emphasize-lines: 12-13,53-58
