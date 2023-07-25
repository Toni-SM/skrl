Multi-agents
============

.. toctree::
    :hidden:

    IPPO <multi_agents/ippo>
    MAPPO <multi_agents/mappo>

Multi-agents are autonomous entities that interact with the environment to learn and improve their behavior. Multi-agents' goal is to learn optimal policies, which are correspondence between states and actions that maximize the cumulative reward received from the environment over time.

.. raw:: html

    <br><hr>

.. list-table::
    :header-rows: 1

    * - Multi-agents
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - :doc:`Independent Proximal Policy Optimization <multi_agents/ippo>` (**IPPO**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Multi-Agent Proximal Policy Optimization <multi_agents/mappo>` (**MAPPO**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

Base class
----------

.. note::

    This is the base class for all multi-agents and provides only basic functionality that is not tied to any implementation of the optimization algorithms.
    **It is not intended to be used directly**.

.. raw:: html

    <br>

Basic inheritance usage
^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. tab:: Inheritance

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../snippets/multi_agent.py
                    :language: python
                    :start-after: [start-multi-agent-base-class-torch]
                    :end-before: [end-multi-agent-base-class-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../snippets/multi_agent.py
                    :language: python
                    :start-after: [start-multi-agent-base-class-jax]
                    :end-before: [end-multi-agent-base-class-jax]

.. raw:: html

    <br>

API (PyTorch)
^^^^^^^^^^^^^

.. autoclass:: skrl.multi_agents.torch.base.MultiAgent
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :private-members: _update, _empty_preprocessor, _get_internal_value, _as_dict
    :members:

    .. automethod:: __init__
    .. automethod:: __str__

.. raw:: html

    <br>

API (JAX)
^^^^^^^^^

.. autoclass:: skrl.multi_agents.jax.base.MultiAgent
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :private-members: _update, _empty_preprocessor, _get_internal_value, _as_dict
    :members:

    .. automethod:: __init__
    .. automethod:: __str__
