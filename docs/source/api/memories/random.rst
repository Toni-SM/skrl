:tocdepth: 4

Random memory
=============

Random sampling memory.

|br| |hr|

Usage
-----

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. literalinclude:: ../../snippets/memories.py
            :language: python
            :emphasize-lines: 2, 7
            :start-after: [start-random-torch]
            :end-before: [end-random-torch]

    .. group-tab:: |_4| |jax| |_4|

        .. literalinclude:: ../../snippets/memories.py
            :language: python
            :emphasize-lines: 2, 7
            :start-after: [start-random-jax]
            :end-before: [end-random-jax]

    .. group-tab:: |_4| |warp| |_4|

        .. literalinclude:: ../../snippets/memories.py
            :language: python
            :emphasize-lines: 2, 7
            :start-after: [start-random-warp]
            :end-before: [end-random-warp]

|

API
---

|

PyTorch
^^^^^^^

.. automodule:: skrl.memories.torch.random
.. autosummary::
    :nosignatures:

    RandomMemory

.. autoclass:: skrl.memories.torch.random.RandomMemory
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __len__

|

Jax
^^^

.. automodule:: skrl.memories.jax.random
.. autosummary::
    :nosignatures:

    RandomMemory

.. autoclass:: skrl.memories.jax.random.RandomMemory
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __len__

|

Warp
^^^^

.. automodule:: skrl.memories.warp.random
.. autosummary::
    :nosignatures:

    RandomMemory

.. autoclass:: skrl.memories.warp.random.RandomMemory
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __len__
