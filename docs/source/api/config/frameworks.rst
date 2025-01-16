ML frameworks configuration
===========================

Configurations for behavior modification of Machine Learning (ML) frameworks.

.. raw:: html

    <br><hr>

PyTorch
-------

PyTorch specific configuration

.. raw:: html

    <br>

API
^^^

.. autofunction:: skrl.config.torch.parse_device

.. py:data:: skrl.config.torch.device
    :type: torch.device
    :value: "cuda:${LOCAL_RANK}" | "cpu"

    Default device.

    The default device, unless specified, is ``cuda:0`` (or ``cuda:LOCAL_RANK`` in a distributed environment) if CUDA is available, ``cpu`` otherwise.

.. py:data:: skrl.config.torch.local_rank
    :type: int
    :value: 0

    The rank of the worker/process (e.g.: GPU) within a local worker group (e.g.: node).

    This property reads from the ``LOCAL_RANK`` environment variable (``0`` if it doesn't exist).
    See `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ for more details.

    Read-only attribute.

.. py:data:: skrl.config.torch.rank
    :type: int
    :value: 0

    The rank of the worker/process (e.g.: GPU) within a worker group (e.g.: across all nodes).

    This property reads from the ``RANK`` environment variable (``0`` if it doesn't exist).
    See `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ for more details.

    Read-only attribute.

.. py:data:: skrl.config.torch.world_size
    :type: int
    :value: 1

    The total number of workers/process (e.g.: GPUs) in a worker group (e.g.: across all nodes).

    This property reads from the ``WORLD_SIZE`` environment variable (``1`` if it doesn't exist).
    See `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`_ for more details.

    Read-only attribute.

.. py:data:: skrl.config.torch.is_distributed
    :type: bool
    :value: False

    Whether if running in a distributed environment.

    This property is ``True`` when the PyTorch's distributed environment variable ``WORLD_SIZE > 1``.

    Read-only attribute.

.. raw:: html

    <br>

JAX
---

JAX specific configuration

.. raw:: html

    <br>

API
^^^

.. autofunction:: skrl.config.jax.parse_device

.. py:data:: skrl.config.jax.device
    :type: jax.Device
    :value: "cuda:${LOCAL_RANK}" | "cpu"

    Default device.

    The default device, unless specified, is ``cuda:0`` (or ``cuda:JAX_LOCAL_RANK`` in a distributed environment) if CUDA is available, ``cpu`` otherwise.

.. py:data:: skrl.config.jax.backend
    :type: str
    :value: "numpy"

    Backend used by the different components to operate and generate arrays.

    This configuration excludes models and optimizers.
    Supported backend are: ``"numpy"`` and ``"jax"``.

.. py:data:: skrl.config.jax.key
    :type: jax.Array
    :value: [0, 0]

    Pseudo-random number generator (PRNG) key.

    Key is formatted as 32-bit unsigned integer and the default device is used.

.. py:data:: skrl.config.jax.local_rank
    :type: int
    :value: 0

    The rank of the worker/process (e.g.: GPU) within a local worker group (e.g.: node).

    This property reads from the ``JAX_LOCAL_RANK`` environment variable (``0`` if it doesn't exist).

    Read-only attribute.

.. py:data:: skrl.config.jax.rank
    :type: int
    :value: 0

    The rank of the worker/process (e.g.: GPU) within a worker group (e.g.: across all nodes).

    This property reads from the ``JAX_RANK`` environment variable (``0`` if it doesn't exist).

    Read-only attribute.

.. py:data:: skrl.config.jax.world_size
    :type: int
    :value: 1

    The total number of workers/process (e.g.: GPUs) in a worker group (e.g.: across all nodes).

    This property reads from the ``JAX_WORLD_SIZE`` environment variable (``1`` if it doesn't exist).

    Read-only attribute.

.. py:data:: skrl.config.jax.coordinator_address
    :type: str
    :value: "127.0.0.1:1234"

    IP address and port where process 0 will start a JAX service.

    This property reads from the ``JAX_COORDINATOR_ADDR:JAX_COORDINATOR_PORT`` environment variables (``127.0.0.1:1234`` if they don't exist).

    Read-only attribute.

.. py:data:: skrl.config.jax.is_distributed
    :type: bool
    :value: False

    Whether if running in a distributed environment.

    This property is ``True`` when the JAX's distributed environment variable ``WORLD_SIZE > 1``.

    Read-only attribute.
