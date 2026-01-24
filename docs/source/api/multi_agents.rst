:tocdepth: 5

Multi-agents
============

.. toctree::
    :hidden:

    IPPO <multi_agents/ippo>
    MAPPO <multi_agents/mappo>

Multi-agents are autonomous entities that interact with the environment to learn and improve their behavior.
Multi-agents' goal is to learn optimal policies, which are correspondence between states and actions that maximize
the cumulative reward received from the environment over time.

|br| |hr|

Implemented agents
------------------

The following table lists the implemented multi-agents and their support for different frameworks.

.. list-table::
    :header-rows: 1

    * - Multi-agents
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`Independent Proximal Policy Optimization <multi_agents/ippo>` (**IPPO**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Multi-Agent Proximal Policy Optimization <multi_agents/mappo>` (**MAPPO**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

|br| |hr|

Base class / configuration
--------------------------

Base class and configuration for multi-agent implementations.

API
^^^

|

PyTorch
"""""""

.. automodule:: skrl.multi_agents.torch
.. autosummary::
    :nosignatures:

    MultiAgentCfg
    ExperimentCfg
    MultiAgent

.. autoclass:: skrl.multi_agents.torch.MultiAgentCfg
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.multi_agents.torch.ExperimentCfg
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.multi_agents.torch.MultiAgent
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

JAX
"""

.. automodule:: skrl.multi_agents.jax
.. autosummary::
    :nosignatures:

    MultiAgentCfg
    ExperimentCfg
    MultiAgent

.. autoclass:: skrl.multi_agents.jax.MultiAgentCfg
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.multi_agents.jax.ExperimentCfg
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.multi_agents.jax.MultiAgent
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
