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

Network definitions
-------------------

The network is composed of one or more containers.
For each container its input, hidden layers and activation functions are specified.

Implementation details:

- The network compute/forward is done by calling the containers in the order in which they are defined
- Containers use :py:class:`torch.nn.Sequential` in PyTorch, and :py:class:`flax.linen.Sequential` in JAX
- If a single activation function is specified (mapping or sequence), it will be applied after each layer (except ``flatten`` layers) in the container

.. tabs::

    .. group-tab:: YAML

        .. literalinclude:: ../../snippets/model_instantiators.yaml
            :language: yaml
            :start-after: [start-structure-yaml]
            :end-before: [end-structure-yaml]

|

Inputs
^^^^^^

The input can be specified using the enum ``Shape`` (see :py:class:`skrl.utils.model_instantiators.torch.Shape`) or previously defined container names.
Certain operations could be specified on them, including indexing (by a range of numbers in sequences, by key in dictionaries) and slicing

.. list-table::
    :header-rows: 1

    * - Operations
      - Example
    * - Concatenation
      - ``concatenate(features_extractor, ACTIONS)``
    * - Indexing
      - ``OBSERVATIONS["camera"]``
        |br| ``OBSERVATIONS[:, 2:5]``
    * - Arithmetic (``+``, ``-``, ``*``, ``/``)
      - ``features_extractor + OBSERVATIONS``

|

Activation functions
^^^^^^^^^^^^^^^^^^^^

The following table lists the supported activation functions:

.. list-table::
    :header-rows: 1

    * - Activations
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - ``relu``
      - :py:class:`torch.nn.ReLU`
      - :py:obj:`flax.linen.activation.relu`
    * - ``tanh``
      - :py:class:`torch.nn.Tanh`
      - :py:obj:`flax.linen.activation.tanh`
    * - ``sigmoid``
      - :py:class:`torch.nn.Sigmoid`
      - :py:obj:`flax.linen.activation.sigmoid`
    * - ``leaky_relu``
      - :py:class:`torch.nn.LeakyReLU`
      - :py:obj:`flax.linen.activation.leaky_relu`
    * - ``elu``
      - :py:class:`torch.nn.ELU`
      - :py:obj:`flax.linen.activation.elu`
    * - ``softplus``
      - :py:class:`torch.nn.Softplus`
      - :py:obj:`flax.linen.activation.softplus`
    * - ``softsign``
      - :py:class:`torch.nn.Softsign`
      - :py:obj:`flax.linen.activation.soft_sign`
    * - ``selu``
      - :py:class:`torch.nn.SELU`
      - :py:obj:`flax.linen.activation.selu`
    * - ``softmax``
      - :py:class:`torch.nn.Softmax`
      - :py:obj:`flax.linen.activation.softmax`

|

Layers
^^^^^^

The following table lists the supported layers and transformations:

.. list-table::
    :header-rows: 1

    * - Layers
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - ``linear``
      - :py:class:`torch.nn.Linear`
      - :py:class:`flax.linen.Dense`
    * - ``conv2d``
      - :py:class:`torch.nn.Conv2d`
      - :py:class:`flax.linen.Conv`
    * - ``flatten``
      - :py:class:`torch.nn.Flatten`
      - :py:obj:`jax.numpy.reshape`

|

linear
""""""

Apply a linear transformation (:py:class:`torch.nn.Linear` in PyTorch, :py:class:`flax.linen.Dense` in JAX)

.. note::

    PyTorch's ``in_features`` parameter size is inferred

.. list-table::
    :header-rows: 1

    * -
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - Type
      - Required
      - Description
    * - 0
      - ``in_features``
      - .. centered:: -
      - ``int``
      - .. centered:: :math:`\blacksquare`
      - Number of input features
    * - 1
      - ``out_features``
      - ``features``
      - ``int``
      - .. centered:: :math:`\blacksquare`
      - Number of output features
    * - 2
      - ``bias``
      - ``use_bias``
      - ``bool``
      - .. centered:: :math:`\square`
      - Whether to add a bias

.. tabs::

    .. group-tab:: YAML

        .. tabs::

            .. group-tab:: Single value

                .. literalinclude:: ../../snippets/model_instantiators.yaml
                    :language: yaml
                    :start-after: [start-layer-linear-single]
                    :end-before: [end-layer-linear-single]

            .. group-tab:: As list

                .. literalinclude:: ../../snippets/model_instantiators.yaml
                    :language: yaml
                    :start-after: [start-layer-linear-list]
                    :end-before: [end-layer-linear-list]

            .. group-tab:: As dict

                .. literalinclude:: ../../snippets/model_instantiators.yaml
                    :language: yaml
                    :start-after: [start-layer-linear-dict]
                    :end-before: [end-layer-linear-dict]

|

conv2d
""""""

Apply a 2D convolution (:py:class:`torch.nn.Conv2d` in PyTorch, :py:class:`flax.linen.Conv` in JAX)

.. note::

    PyTorch's ``in_channels`` parameter value is inferred from input or previous layer

.. list-table::
    :header-rows: 1

    * -
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - Type
      - Required
      - Description
    * - 0
      - ``in_channels``
      - .. centered:: -
      - ``int``
      - .. centered:: :math:`\blacksquare`
      - Number of input channels
    * - 1
      - ``out_channels``
      - ``features``
      - ``int``
      - .. centered:: :math:`\blacksquare`
      - Number of output channels (filters)
    * - 2
      - ``kernel_size``
      - ``kernel_size``
      - ``int``, ``tuple[int]``
      - .. centered:: :math:`\blacksquare`
      - Convolutional kernel size
    * - 3
      - ``stride``
      - ``strides``
      - ``int``, ``tuple[int]``
      - .. centered:: :math:`\square`
      - Inter-window strides
    * - 4
      - ``padding``
      - ``padding``
      - ``str``, ``int``, ``tuple[int]``
      - .. centered:: :math:`\square`
      - Padding added to all dimensions
    * - 5
      - ``bias``
      - ``use_bias``
      - ``bool``
      - .. centered:: :math:`\square`
      - Whether to add a bias

.. tabs::

    .. group-tab:: YAML

        .. tabs::

            .. group-tab:: As list

                .. literalinclude:: ../../snippets/model_instantiators.yaml
                    :language: yaml
                    :start-after: [start-layer-conv-list]
                    :end-before: [end-layer-conv-list]

            .. group-tab:: As dict

                .. literalinclude:: ../../snippets/model_instantiators.yaml
                    :language: yaml
                    :start-after: [start-layer-conv-dict]
                    :end-before: [end-layer-conv-dict]

|

flatten
"""""""

Flatten a contiguous range of dimensions (:py:class:`torch.nn.Flatten` in PyTorch, :py:obj:`jax.numpy.reshape` operation in JAX)

.. list-table::
    :header-rows: 1

    * -
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - Type
      - Required
      - Description
    * - 0
      - ``start_dim``
      - .. centered:: -
      - ``int``
      - .. centered:: :math:`\square`
      - First dimension to flatten
    * - 1
      - ``end_dim``
      - .. centered:: -
      - ``int``
      - .. centered:: :math:`\square`
      - Last dimension to flatten

.. tabs::

    .. group-tab:: YAML

        .. tabs::

            .. group-tab:: Single value

                .. literalinclude:: ../../snippets/model_instantiators.yaml
                    :language: yaml
                    :start-after: [start-layer-flatten-single]
                    :end-before: [end-layer-flatten-single]

            .. group-tab:: As list

                .. literalinclude:: ../../snippets/model_instantiators.yaml
                    :language: yaml
                    :start-after: [start-layer-flatten-list]
                    :end-before: [end-layer-flatten-list]

            .. group-tab:: As dict

                .. literalinclude:: ../../snippets/model_instantiators.yaml
                    :language: yaml
                    :start-after: [start-layer-flatten-dict]
                    :end-before: [end-layer-flatten-dict]

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
