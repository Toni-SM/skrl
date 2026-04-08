:tocdepth: 5

Memories
========

.. toctree::
    :hidden:

    Random <memories/random>

Memories are storage components that allow agents to collect and use/reuse current or past experiences of their
interaction with the environment or other types of information.

|br| |hr|

Implemented memories
--------------------

The following table lists the implemented memories and their support for different frameworks.

.. list-table::
    :header-rows: 1

    * - Memories
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`Random memory <memories/random>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

|br| |hr|

Base class
----------

Base class for memories.

API
^^^

|

PyTorch
"""""""

.. automodule:: skrl.memories.torch
.. autosummary::
    :nosignatures:

    Memory

.. autoclass:: skrl.memories.torch.Memory
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __len__

|

JAX
"""

.. automodule:: skrl.memories.jax
.. autosummary::
    :nosignatures:

    Memory

.. autoclass:: skrl.memories.jax.Memory
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __len__

|

Warp
""""

.. automodule:: skrl.memories.warp
.. autosummary::
    :nosignatures:

    Memory

.. autoclass:: skrl.memories.warp.Memory
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __len__
