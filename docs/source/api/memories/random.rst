Random memory
=============

Random sampling memory

.. raw:: html

    <br><hr>

Usage
-----

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. literalinclude:: ../../snippets/memories.py
            :language: python
            :emphasize-lines: 2, 5
            :start-after: [start-random-torch]
            :end-before: [end-random-torch]

    .. group-tab:: |_4| |jax| |_4|

        .. literalinclude:: ../../snippets/memories.py
            :language: python
            :emphasize-lines: 2, 5
            :start-after: [start-random-jax]
            :end-before: [end-random-jax]

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.memories.torch.random.RandomMemory
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__
    .. automethod:: __len__

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.memories.jax.random.RandomMemory
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__
    .. automethod:: __len__
