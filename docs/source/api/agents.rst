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
