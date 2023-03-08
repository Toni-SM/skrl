Multi-agents
============

.. toctree::
    :hidden:

    IPPO <multi_agents/ippo>
    MAPPO <multi_agents/mappo>

Agents are autonomous entities that interact with the environment to learn and improve their behavior. Multi-agents' goal is to learn optimal policies, which are correspondence between states and actions that maximize the cumulative reward received from the environment over time.

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

        .. literalinclude:: ../snippets/multi_agent.py
            :language: python

.. raw:: html

    <br>

API
^^^

.. autoclass:: skrl.multi_agents.torch.base.MultiAgent
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :private-members: _update, _empty_preprocessor, _get_internal_value, _as_dict
    :members:

    .. automethod:: __init__
    .. automethod:: __str__
