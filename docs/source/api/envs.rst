Environments
============

.. toctree::
    :hidden:

    Wrapping (single-agent) <envs/wrapping>
    Wrapping (multi-agents) <envs/multi_agents_wrapping>
    Isaac Gym environments <envs/isaac_gym>
    Isaac Orbit environments <envs/isaac_orbit>
    Omniverse Isaac Gym environments <envs/omniverse_isaac_gym>

The environment plays a fundamental and crucial role in defining the RL setup. It is the place where the agent interacts, and it is responsible for providing the agent with information about its current state, as well as the rewards/penalties associated with each action.

.. raw:: html

    <br><hr>

Grouped in this section you will find how to load environments from NVIDIA Isaac Gym, Isaac Orbit and Omniverse Isaac Gym with a simple function.

In addition, you will be able to :doc:`wrap single-agent <envs/wrapping>` and :doc:`multi-agent <envs/multi_agents_wrapping>` RL environment interfaces.

.. list-table::
    :header-rows: 1

    * - Loaders
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - :doc:`Isaac Gym environments <envs/isaac_gym>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Isaac Orbit environments <envs/isaac_orbit>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Omniverse Isaac Gym environments <envs/omniverse_isaac_gym>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

.. list-table::
    :header-rows: 1

    * - Wrappers
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - Bi-DexHands
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - DeepMind
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - Gym
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - Gymnasium
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - Isaac Gym (previews)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - Isaac Orbit
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - Omniverse Isaac Gym |_5| |_5| |_5| |_5| |_2|
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - PettingZoo
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - robosuite
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - Shimmy
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
