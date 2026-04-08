:tocdepth: 5

Models
======

.. toctree::
    :hidden:

    Tabular <models/tabular>
    Categorical <models/categorical>
    Multi-Categorical <models/multicategorical>
    Gaussian <models/gaussian>
    Multivariate Gaussian <models/multivariate_gaussian>
    Deterministic <models/deterministic>
    Shared model <models/shared_model>

Models (or agent models) refer to a representation of the agent's policy, value function, etc. that the agent uses
to make decisions. Agents can have one or more models, and their parameters are adjusted by the optimization algorithms.

|br| |hr|

Implemented models
------------------

The following table lists the implemented models and their support for different frameworks.

.. list-table::
    :header-rows: 1

    * - Models
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`Tabular model <models/tabular>` (discrete domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Categorical model <models/categorical>` (discrete domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Multi-Categorical model <models/multicategorical>` (discrete domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Gaussian model <models/gaussian>` (continuous domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Multivariate Gaussian model <models/multivariate_gaussian>` (continuous domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Deterministic model <models/deterministic>` (continuous domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Shared model <models/shared_model>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\blacksquare`

|br| |hr|

.. _models_base_class:

Base class
----------

Base class for models.

API
^^^

|

PyTorch
"""""""

.. automodule:: skrl.models.torch
.. autosummary::
    :nosignatures:

    Model

.. autoclass:: skrl.models.torch.Model
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

JAX
"""

.. automodule:: skrl.models.jax
.. autosummary::
    :nosignatures:

    Model

.. autoclass:: skrl.models.jax.Model
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

Warp
""""

.. automodule:: skrl.models.warp
.. autosummary::
    :nosignatures:

    Model

.. autoclass:: skrl.models.warp.Model
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
