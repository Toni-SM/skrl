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
    * - :doc:`Multi-categorical model <../models/multicategorical>` (discrete domain)
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

        .. literalinclude:: ../../snippets/model_instantiators.txt
            :language: yaml
            :start-after: [start-structure-yaml]
            :end-before: [end-structure-yaml]

    .. group-tab:: Python

        .. literalinclude:: ../../snippets/model_instantiators.txt
            :language: python
            :start-after: [start-structure-python]
            :end-before: [end-structure-python]

|

Inputs
^^^^^^

Inputs can be specified using tokens or previously defined container outputs (by container name).
Certain operations could be specified on them, including indexing and slicing

.. hint::

    Operations can be mixed to create complex input statements

Available tokens:

* ``OBSERVATIONS``: Unflattened tensorized input ``observations`` (tensorized observation space) forwarded to the model
* ``STATES``: Unflattened tensorized input ``states`` (tensorized state space) forwarded to the model
* ``ACTIONS``: Unflattened tensorized input ``taken_actions`` (tensorized action space) forwarded to the model

Supported operations:

.. list-table::
    :header-rows: 1

    * - Operations
      - Example
    * - Tensor/array indexing and slicing
        |br| E.g.: :py:class:`gymnasium.spaces.Box`
      - ``OBSERVATIONS[:, 0]``
        |br| ``OBSERVATIONS[:, 2:5]``
    * - Dictionary indexing by key
        |br| E.g.: :py:class:`gymnasium.spaces.Dict`
      - ``STATES["joint-pos"]``
    * - Arithmetic (``+``, ``-``, ``*``, ``/``)
      - ``features_extractor + ACTIONS``
    * - Concatenation
      - ``concatenate([features_extractor, ACTIONS])``
    * - Permute dimensions
      - ``permute(STATES, (0, 3, 1, 2))``

|

Output
^^^^^^

The output can be specified using tokens or defined container outputs (by container name).
Certain operations could be specified on it

.. note::

    If a token is used, a linear layer will be created with the last container in the list (as the number of input features)
    and the value represented by the token (as the number of output features)

.. hint::

    Operations can be mixed to create complex output statement

Available tokens:

* ``ACTIONS``: Token indicating that the output shape is the number of elements in the action space
* ``ONE``: Token indicating that the output shape is 1

Supported operations:

.. list-table::
    :header-rows: 1

    * - Operations
      - Example
    * - Activation function
      - ``tanh(ACTIONS)``
    * - Arithmetic (``+``, ``-``, ``*``, ``/``)
      - ``features_extractor + ONE``
    * - Concatenation
      - ``concatenate([features_extractor, net])``

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

    The tokens ``NUM_OBSERVATIONS`` (number of elements in the observation space), ``NUM_STATES`` (number of elements in the state space),
    ``NUM_ACTIONS`` (number of elements in the action space), and ``ONE`` (1) can be used as the layer's number of input/output features

.. note::

    If the PyTorch's ``in_features`` parameter is not specified it will be inferred by using the :py:class:`torch.nn.LazyLinear` module

.. list-table::
    :header-rows: 1

    * -
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - Type
      - Required
      - Description
    * -
      - ``in_features``
      - .. centered:: -
      - ``int``
      - .. centered:: :math:`\square`
      - Number of input features
    * - 0
      - ``out_features``
      - ``features``
      - ``int``
      - .. centered:: :math:`\blacksquare`
      - Number of output features
    * - 1
      - ``bias``
      - ``use_bias``
      - ``bool``
      - .. centered:: :math:`\square`
      - Whether to add a bias

.. tabs::

    .. group-tab:: YAML

        .. tabs::

            .. group-tab:: Single value

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: yaml
                    :start-after: [start-layer-linear-basic]
                    :end-before: [end-layer-linear-basic]

            .. group-tab:: As int

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: yaml
                    :start-after: [start-layer-linear-int]
                    :end-before: [end-layer-linear-int]

            .. group-tab:: As list

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: yaml
                    :start-after: [start-layer-linear-list]
                    :end-before: [end-layer-linear-list]

            .. group-tab:: As dict

                .. hint::

                    The parameter names can be interchanged/mixed between PyTorch and JAX

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: yaml
                    :start-after: [start-layer-linear-dict]
                    :end-before: [end-layer-linear-dict]

    .. group-tab:: Python

        .. tabs::

            .. group-tab:: Single value

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: python
                    :start-after: [start-layer-linear-basic-python]
                    :end-before: [end-layer-linear-basic-python]

            .. group-tab:: As int

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: python
                    :start-after: [start-layer-linear-int-python]
                    :end-before: [end-layer-linear-int-python]

            .. group-tab:: As list

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: python
                    :start-after: [start-layer-linear-list-python]
                    :end-before: [end-layer-linear-list-python]

            .. group-tab:: As dict

                .. hint::

                    The parameter names can be interchanged/mixed between PyTorch and JAX

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: python
                    :start-after: [start-layer-linear-dict-python]
                    :end-before: [end-layer-linear-dict-python]

|

conv2d
""""""

Apply a 2D convolution (:py:class:`torch.nn.Conv2d` in PyTorch, :py:class:`flax.linen.Conv` in JAX)

.. warning::

    * PyTorch :py:class:`torch.nn.Conv2d` expects the input to be in the form NCHW (*N*: batch, *C*: channels, *H*: height, *W*: width).
      A permutation operation may be necessary to modify the dimensions of a batch of images which are typically NHWC.
    * JAX :py:class:`flax.linen.Conv` expects the input to be in the form NHWC (the typical dimensions of a batch of images).

.. note::

    If the PyTorch's ``in_channels`` parameter is not specified it will be inferred by using the :py:class:`torch.nn.LazyConv2d` module

.. list-table::
    :header-rows: 1

    * -
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - Type
      - Required
      - Description
    * -
      - ``in_channels``
      - .. centered:: -
      - ``int``
      - .. centered:: :math:`\square`
      - Number of input channels
    * - 0
      - ``out_channels``
      - ``features``
      - ``int``
      - .. centered:: :math:`\blacksquare`
      - Number of output channels (filters)
    * - 1
      - ``kernel_size``
      - ``kernel_size``
      - ``int``, ``tuple[int]``
      - .. centered:: :math:`\blacksquare`
      - Convolutional kernel size
    * - 2
      - ``stride``
      - ``strides``
      - ``int``, ``tuple[int]``
      - .. centered:: :math:`\square`
      - Inter-window strides
    * - 3
      - ``padding``
      - ``padding``
      - ``str``, ``int``, ``tuple[int]``
      - .. centered:: :math:`\square`
      - Padding added to all dimensions
    * - 4
      - ``bias``
      - ``use_bias``
      - ``bool``
      - .. centered:: :math:`\square`
      - Whether to add a bias

.. tabs::

    .. group-tab:: YAML

        .. tabs::

            .. group-tab:: As list

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: yaml
                    :start-after: [start-layer-conv2d-list]
                    :end-before: [end-layer-conv2d-list]

            .. group-tab:: As dict

                .. hint::

                    The parameter names can be interchanged/mixed between PyTorch and JAX

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: yaml
                    :start-after: [start-layer-conv2d-dict]
                    :end-before: [end-layer-conv2d-dict]

    .. group-tab:: Python

        .. tabs::

            .. group-tab:: As list

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: python
                    :start-after: [start-layer-conv2d-list-python]
                    :end-before: [end-layer-conv2d-list-python]

            .. group-tab:: As dict

                .. hint::

                    The parameter names can be interchanged/mixed between PyTorch and JAX

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: python
                    :start-after: [start-layer-conv2d-dict-python]
                    :end-before: [end-layer-conv2d-dict-python]

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

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: yaml
                    :start-after: [start-layer-flatten-basic]
                    :end-before: [end-layer-flatten-basic]

            .. group-tab:: As list

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: yaml
                    :start-after: [start-layer-flatten-list]
                    :end-before: [end-layer-flatten-list]

            .. group-tab:: As dict

                .. hint::

                    The parameter names can be interchanged/mixed between PyTorch and JAX

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: yaml
                    :start-after: [start-layer-flatten-dict]
                    :end-before: [end-layer-flatten-dict]

    .. group-tab:: Python

        .. tabs::

            .. group-tab:: Single value

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: python
                    :start-after: [start-layer-flatten-basic-python]
                    :end-before: [end-layer-flatten-basic-python]

            .. group-tab:: As list

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: python
                    :start-after: [start-layer-flatten-list-python]
                    :end-before: [end-layer-flatten-list-python]

            .. group-tab:: As dict

                .. hint::

                    The parameter names can be interchanged/mixed between PyTorch and JAX

                .. literalinclude:: ../../snippets/model_instantiators.txt
                    :language: python
                    :start-after: [start-layer-flatten-dict-python]
                    :end-before: [end-layer-flatten-dict-python]

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autofunction:: skrl.utils.model_instantiators.torch.categorical_model

.. autofunction:: skrl.utils.model_instantiators.torch.multicategorical_model

.. autofunction:: skrl.utils.model_instantiators.torch.deterministic_model

.. autofunction:: skrl.utils.model_instantiators.torch.gaussian_model

.. autofunction:: skrl.utils.model_instantiators.torch.multivariate_gaussian_model

.. autofunction:: skrl.utils.model_instantiators.torch.shared_model

.. raw:: html

    <br>

API (JAX)
---------

.. autofunction:: skrl.utils.model_instantiators.jax.categorical_model

.. autofunction:: skrl.utils.model_instantiators.jax.deterministic_model

.. autofunction:: skrl.utils.model_instantiators.jax.gaussian_model
