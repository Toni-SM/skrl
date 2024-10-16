Spaces
======

Utilities to operate on Gymnasium `spaces <https://gymnasium.farama.org/api/spaces>`_.

.. raw:: html

    <br><hr>

Overview
--------

The utilities described in this section supports the following Gymnasium spaces:

.. list-table::
    :header-rows: 1

    * - Type
      - Supported spaces
    * - Fundamental
      - :py:class:`~gymnasium.spaces.Box`, :py:class:`~gymnasium.spaces.Discrete`, and :py:class:`~gymnasium.spaces.MultiDiscrete`
    * - Composite
      - :py:class:`~gymnasium.spaces.Dict` and :py:class:`~gymnasium.spaces.Tuple`

The following table provides a snapshot of the space sample conversion functions:

.. list-table::
    :header-rows: 1

    * - Input
      - Function
      - Output
    * - Space (NumPy / int)
      - :py:func:`~skrl.utils.spaces.torch.tensorize_space`
      - Space (PyTorch / JAX)
    * - Space (PyTorch / JAX)
      - :py:func:`~skrl.utils.spaces.torch.untensorize_space`
      - Space (NumPy / int)
    * - Space (PyTorch / JAX)
      - :py:func:`~skrl.utils.spaces.torch.flatten_tensorized_space`
      - PyTorch tensor / JAX array
    * - PyTorch tensor / JAX array
      - :py:func:`~skrl.utils.spaces.torch.unflatten_tensorized_space`
      - Space (PyTorch / JAX)

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autofunction:: skrl.utils.spaces.torch.compute_space_size

.. autofunction:: skrl.utils.spaces.torch.convert_gym_space

.. autofunction:: skrl.utils.spaces.torch.flatten_tensorized_space

.. autofunction:: skrl.utils.spaces.torch.sample_space

.. autofunction:: skrl.utils.spaces.torch.tensorize_space

.. autofunction:: skrl.utils.spaces.torch.unflatten_tensorized_space

.. autofunction:: skrl.utils.spaces.torch.untensorize_space

.. raw:: html

    <br>

API (JAX)
---------

.. autofunction:: skrl.utils.spaces.jax.compute_space_size

.. autofunction:: skrl.utils.spaces.jax.convert_gym_space

.. autofunction:: skrl.utils.spaces.jax.flatten_tensorized_space

.. autofunction:: skrl.utils.spaces.jax.sample_space

.. autofunction:: skrl.utils.spaces.jax.tensorize_space

.. autofunction:: skrl.utils.spaces.jax.unflatten_tensorized_space

.. autofunction:: skrl.utils.spaces.jax.untensorize_space
