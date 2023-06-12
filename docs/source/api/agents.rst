Agents
======

.. toctree::
    :hidden:

    A2C <agents/a2c>
    AMP <agents/amp>
    CEM <agents/cem>
    DDPG <agents/ddpg>
    DDQN <agents/ddqn>
    DQN <agents/dqn>
    PPO <agents/ppo>
    Q-learning <agents/q_learning>
    RPO <agents/rpo>
    SAC <agents/sac>
    SARSA <agents/sarsa>
    TD3 <agents/td3>
    TRPO <agents/trpo>

Agents are autonomous entities that interact with the environment to learn and improve their behavior. Agents' goal is to learn an optimal policy, which is a correspondence between states and actions that maximizes the cumulative reward received from the environment over time.

.. raw:: html

    <br><hr>

.. list-table::
    :header-rows: 1

    * - Agents
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - :doc:`Advantage Actor Critic <agents/a2c>` (**A2C**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Adversarial Motion Priors <agents/amp>` (**AMP**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Cross-Entropy Method <agents/cem>` (**CEM**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Deep Deterministic Policy Gradient <agents/ddpg>` (**DDPG**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Double Deep Q-Network <agents/ddqn>` (**DDQN**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Deep Q-Network <agents/dqn>` (**DQN**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Proximal Policy Optimization <agents/ppo>` (**PPO**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Q-learning <agents/q_learning>` (**Q-learning**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Robust Policy Optimization <agents/rpo>` (**RPO**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Soft Actor-Critic <agents/sac>` (**SAC**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`State Action Reward State Action <agents/sarsa>` (**SARSA**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Twin-Delayed DDPG <agents/td3>` (**TD3**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Trust Region Policy Optimization <agents/trpo>` (**TRPO**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

Base class
----------

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.

.. raw:: html

    <br>

Basic inheritance usage
^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. tab:: Inheritance

        .. literalinclude:: ../snippets/agent.py
            :language: python

.. raw:: html

    <br>

API
^^^

.. autoclass:: skrl.agents.torch.base.Agent
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :private-members: _update, _empty_preprocessor, _get_internal_value
    :members:

    .. automethod:: __init__
    .. automethod:: __str__
