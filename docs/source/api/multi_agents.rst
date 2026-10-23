:tocdepth: 5

Multi-agents
============

.. toctree::
    :hidden:

    HAA2C <multi_agents/haa2c>
    HAD3QN <multi_agents/had3qn>
    HADDPG <multi_agents/haddpg>
    HAPPO <multi_agents/happo>
    HASAC <multi_agents/hasac>
    HATD3 <multi_agents/hatd3>
    HATRPO <multi_agents/hatrpo>
    IPPO <multi_agents/ippo>
    MADDPG <multi_agents/maddpg>
    MAPPO <multi_agents/mappo>
    MATD3 <multi_agents/matd3>

Multi-agents are autonomous entities that interact with the environment to learn and improve their behavior.
Multi-agents' goal is to learn optimal policies, which are correspondence between states and actions that maximize
the cumulative reward received from the environment over time.

|br| |hr|

Implemented multi-agents
------------------------

The following table lists the implemented multi-agents and their support for different frameworks.

.. list-table::
    :header-rows: 1

    * - Multi-agents
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`Heterogeneous-Agent Advantage Actor-Critic <multi_agents/haa2c>` (**HAA2C**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Heterogeneous-Agent Dueling Double Deep Q-Network <multi_agents/had3qn>` (**HAD3QN**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Heterogeneous-Agent Deep Deterministic Policy Gradient <multi_agents/haddpg>` (**HADDPG**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Heterogeneous-Agent Proximal Policy Optimization <multi_agents/happo>` (**HAPPO**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Heterogeneous-Agent Soft Actor-Critic <multi_agents/hasac>` (**HASAC**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Heterogeneous-Agent Twin-Delayed DDPG <multi_agents/hatd3>` (**HATD3**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Heterogeneous-Agent Trust Region Policy Optimization <multi_agents/hatrpo>` (**HATRPO**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Independent Proximal Policy Optimization <multi_agents/ippo>` (**IPPO**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Multi-Agent Deep Deterministic Policy Gradient <multi_agents/maddpg>` (**MADDPG**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Multi-Agent Proximal Policy Optimization <multi_agents/mappo>` (**MAPPO**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Multi-Agent Twin-Delayed DDPG <multi_agents/matd3>` (**MATD3**)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
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
