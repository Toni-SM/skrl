Model instantiators
===================

Utilities for quickly creating model instances.

.. raw:: html

    <br><hr>

.. TODO: add snippet

.. list-table::
    :header-rows: 1

    * - Models
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - :doc:`Tabular model <../models/tabular>` (discrete domain)
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Categorical model <../models/categorical>` (discrete domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Gaussian model <../models/gaussian>` (continuous domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Multivariate Gaussian model <../models/multivariate_gaussian>` (continuous domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - :doc:`Deterministic model <../models/deterministic>` (continuous domain)
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Shared model <../models/shared_model>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.utils.model_instantiators.torch.Shape

    .. py:property:: ONE

        Flag to indicate that the model's input/output has shape (1,)

        This flag is useful for the definition of critic models, where the critic's output is a scalar

    .. py:property:: STATES

        Flag to indicate that the model's input/output is the state (observation) space of the environment
        It is an alias for :py:attr:`OBSERVATIONS`

    .. py:property:: OBSERVATIONS

        Flag to indicate that the model's input/output is the observation space of the environment

    .. py:property:: ACTIONS

        Flag to indicate that the model's input/output is the action space of the environment

    .. py:property:: STATES_ACTIONS

        Flag to indicate that the model's input/output is the combination (concatenation) of the state (observation) and action spaces of the environment

.. autofunction:: skrl.utils.model_instantiators.torch.categorical_model

.. autofunction:: skrl.utils.model_instantiators.torch.deterministic_model

.. autofunction:: skrl.utils.model_instantiators.torch.gaussian_model

.. autofunction:: skrl.utils.model_instantiators.torch.multivariate_gaussian_model

.. autofunction:: skrl.utils.model_instantiators.torch.shared_model

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.utils.model_instantiators.jax.Shape

    .. py:property:: ONE

        Flag to indicate that the model's input/output has shape (1,)

        This flag is useful for the definition of critic models, where the critic's output is a scalar

    .. py:property:: STATES

        Flag to indicate that the model's input/output is the state (observation) space of the environment
        It is an alias for :py:attr:`OBSERVATIONS`

    .. py:property:: OBSERVATIONS

        Flag to indicate that the model's input/output is the observation space of the environment

    .. py:property:: ACTIONS

        Flag to indicate that the model's input/output is the action space of the environment

    .. py:property:: STATES_ACTIONS

        Flag to indicate that the model's input/output is the combination (concatenation) of the state (observation) and action spaces of the environment

.. autofunction:: skrl.utils.model_instantiators.jax.categorical_model

.. autofunction:: skrl.utils.model_instantiators.jax.deterministic_model

.. autofunction:: skrl.utils.model_instantiators.jax.gaussian_model
