Environments
============

.. toctree::
    :hidden:

    Wrapping (single-agent) <envs/wrapping>
    Wrapping (multi-agents) <envs/multi_agents_wrapping>
    Isaac Lab environments <envs/isaaclab>
    Playground environments <envs/playground>

The environment plays a fundamental and crucial role in defining the RL setup. It is the place where the agent interacts, and it is responsible for providing the agent with information about its current state, as well as the rewards/penalties associated with each action.

.. raw:: html

    <br><hr>

In this section you will find how to load environments from NVIDIA Isaac Lab and MuJoCo Playground with a simple function.

.. list-table::
    :header-rows: 1

    * - Loaders
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`Isaac Lab environments <envs/isaaclab>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Playground environments <envs/playground>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

In addition, you will be able to :doc:`wrap single-agent <envs/wrapping>` and :doc:`multi-agent <envs/multi_agents_wrapping>` RL environment interfaces.

.. list-table::
    :header-rows: 1

    * - Wrappers
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - Gym
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - Gymnasium
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - Isaac Lab
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - ManiSkill
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - PettingZoo
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - Playground
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - Shimmy
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
