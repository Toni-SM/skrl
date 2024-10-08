Spaces
======

Utilities to operate on gymnasium `spaces <https://gymnasium.farama.org/api/spaces>`_.

.. raw:: html

    <br><hr>

.. list-table::
    :header-rows: 1

    * - Input
      - Function
      - Output
    * - Space sample with
        |br| NumPy values
      - :py:func:`~skrl.utils.spaces.torch.tensorize_space`
      - Space sample with
        |br| PyTorch / JAX values
    * - Space sample with
        |br| PyTorch / JAX values
      - :py:func:`~skrl.utils.spaces.torch.untensorize_space`
      - Space sample with
        |br| NumPy values
    * - Space sample with
        |br| PyTorch / JAX values
      - :py:func:`~skrl.utils.spaces.torch.flatten_tensorized_space`
      - PyTorch tensor / JAX array
    * - PyTorch tensor / JAX array
      - :py:func:`~skrl.utils.spaces.torch.unflatten_tensorized_space`
      - Space sample with
        |br| PyTorch / JAX values

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
