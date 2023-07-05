Models
======

.. toctree::
    :hidden:

    Tabular <models/tabular>
    Categorical <models/categorical>
    Gaussian <models/gaussian>
    Multivariate Gaussian <models/multivariate_gaussian>
    Deterministic <models/deterministic>
    Shared model <models/shared_model>

Models (or agent models) refer to a representation of the agent's policy, value function, etc. that the agent uses to make decisions. Agents can have one or more models, and their parameters are adjusted by the optimization algorithms.

.. raw:: html

    <br><hr>

.. list-table::
    :header-rows: 1

    * - Models
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - :doc:`Tabular model <models/tabular>` (discrete domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Categorical model <models/categorical>` (discrete domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Gaussian model <models/gaussian>` (continuous domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Multivariate Gaussian model <models/multivariate_gaussian>` (continuous domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Deterministic model <models/deterministic>` (continuous domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Shared model <models/shared_model>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

Base class
----------

.. note::

    This is the base class for all models in this module and provides only basic functionality that is not tied to any specific implementation.
    **It is not intended to be used directly**.

.. raw:: html

    <br>

Mixin and inheritance
^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. tab:: Mixin

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../snippets/model_mixin.py
                    :language: python
                    :start-after: [start-mixin-torch]
                    :end-before: [end-mixin-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../snippets/model_mixin.py
                    :language: python
                    :start-after: [start-mixin-jax]
                    :end-before: [end-mixin-jax]

    .. tab:: Model inheritance

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../snippets/model_mixin.py
                    :language: python
                    :start-after: [start-model-torch]
                    :end-before: [end-model-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../snippets/model_mixin.py
                    :language: python
                    :start-after: [start-model-jax]
                    :end-before: [end-model-jax]

.. raw:: html

    <br>

.. _models_base_class:

API (PyTorch)
^^^^^^^^^^^^^

.. autoclass:: skrl.models.torch.base.Model
    :undoc-members:
    :show-inheritance:
    :private-members: _get_space_size
    :members:

    .. automethod:: __init__

    .. py:property:: device

        Device to be used for the computations

    .. py:property:: observation_space

        Observation/state space. It is a replica of the class constructor parameter of the same name

    .. py:property:: action_space

        Action space. It is a replica of the class constructor parameter of the same name

    .. py:property:: num_observations

        Number of elements in the observation/state space

    .. py:property:: num_actions

        Number of elements in the action space

.. raw:: html

    <br>

API (JAX)
^^^^^^^^^

.. autoclass:: skrl.models.jax.base.Model
    :undoc-members:
    :show-inheritance:
    :private-members: _get_space_size
    :members:

    .. automethod:: __init__

    .. py:property:: device

        Device to be used for the computations

    .. py:property:: observation_space

        Observation/state space. It is a replica of the class constructor parameter of the same name

    .. py:property:: action_space

        Action space. It is a replica of the class constructor parameter of the same name

    .. py:property:: num_observations

        Number of elements in the observation/state space

    .. py:property:: num_actions

        Number of elements in the action space
