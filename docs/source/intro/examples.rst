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

.. literalinclude:: ../examples/isaacgym_single.py
    :language: python
    :linenos:
    :emphasize-lines: 12-13,53-58

Learning in a Isaac Gym environment (parallel agents, multiple environments)
----------------------------------------------------------------------------

This example performs the training of 3 agents by scopes in Isaac Gym's Cartpole environment in the same run. It tries to load the environment from preview 3, but if it fails, it will try to load the environment from preview 2

.. image:: ../_static/imgs/example_parallel.jpg
      :width: 100%
      :align: center
      :alt: Simultaneous training

Two versions are presented:

- Simultaneous training of agents **sharing the same memory** and whose scopes are automatically selected as equally as possible

- Simultaneous training of agents **with individual memory** (no memory sharing) and whose scopes are manually specified and differ from each other

.. tabs::
            
    .. tab:: Shared memory

        .. literalinclude:: ../examples/isaacgym_parallel_shared_memory.py
            :language: python
            :linenos:
            :emphasize-lines: 81,149,156,163

    .. tab:: No shared memory

        .. literalinclude:: ../examples/isaacgym_parallel_no_shared_memory.py
            :language: python
            :linenos:
            :emphasize-lines: 81-83,151,158,165,177
