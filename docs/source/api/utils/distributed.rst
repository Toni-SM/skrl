Distributed
===========

Utilities to start multiple processes from a single program invocation in distributed learning

.. raw:: html

    <br><hr>

PyTorch
-------

PyTorch provides a Python module/console script to launch distributed runs. Visit PyTorch's `torchrun <https://pytorch.org/docs/stable/elastic/run.html>`_ documentation for more details.

The following environment variables available in all processes can be accessed through the library:

* ``LOCAL_RANK`` (accessible via :data:`skrl.config.torch.local_rank`): The local rank.
* ``RANK`` (accessible via :data:`skrl.config.torch.rank`): The global rank.
* ``WORLD_SIZE`` (accessible via :data:`skrl.config.torch.world_size`): The world size (total number of workers in the job).

JAX
---

According to the JAX documentation for `multi-host and multi-process environments <https://jax.readthedocs.io/en/latest/multi_process.html#launching-jax-processes>`_, JAX doesn't automatically start multiple processes from a single program invocation.

Therefore, in order to make distributed learning simpler, this library provides a module (based on the PyTorch ``torch.distributed.run`` module) for launching multi-host and multi-process learning  directly from the command line.

This module launches, in multiple processes, the same JAX Python program (Single Program, Multiple Data (SPMD) parallel computation technique) that defines the following environment variables for each process:

* ``JAX_LOCAL_RANK`` (accessible via :data:`skrl.config.jax.local_rank`): The rank of the worker/process (e.g.: GPU) within a local worker group (e.g.: node).
* ``JAX_RANK`` (accessible via :data:`skrl.config.jax.rank`): The rank (ID number) of the worker/process (e.g.: GPU) within a worker group (e.g.: across all nodes).
* ``JAX_WORLD_SIZE`` (accessible via :data:`skrl.config.jax.world_size`): The total number of workers/process (e.g.: GPUs) in a worker group (e.g.: across all nodes).
* ``JAX_COORDINATOR_ADDR`` (accessible via :data:`skrl.config.jax.coordinator_address`): IP address where process 0 will start a JAX coordinator service.
* ``JAX_COORDINATOR_PORT`` (accessible via :data:`skrl.config.jax.coordinator_address`): Port where process 0 will start a JAX coordinator service.

.. raw:: html

    <br>

Usage
^^^^^

.. code-block:: bash

    $ python -m skrl.utils.distributed.jax --help

.. literalinclude:: ../../snippets/utils_distributed.txt
    :language: text
    :start-after: [start-distributed-launcher-jax]
    :end-before: [end-distributed-launcher-jax]

.. raw:: html

    <br>

API
^^^

.. autofunction:: skrl.utils.distributed.jax.launcher.launch
