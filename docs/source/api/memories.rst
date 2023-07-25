Memories
========

.. toctree::
    :hidden:

    Random <memories/random>

Memories are storage components that allow agents to collect and use/reuse current or past experiences of their interaction with the environment or other types of information.

.. raw:: html

    <br><hr>

.. list-table::
    :header-rows: 1

    * - Memories
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - :doc:`Random memory <memories/random>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

Base class
----------

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.

.. raw:: html

    <br>

Basic inheritance usage
^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. literalinclude:: ../snippets/memories.py
            :language: python
            :start-after: [start-base-class-torch]
            :end-before: [end-base-class-torch]

    .. group-tab:: |_4| |jax| |_4|

        .. literalinclude:: ../snippets/memories.py
            :language: python
            :start-after: [start-base-class-jax]
            :end-before: [end-base-class-jax]

.. raw:: html

    <br>

API (PyTorch)
^^^^^^^^^^^^^

.. autoclass:: skrl.memories.torch.base.Memory
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__
    .. automethod:: __len__

.. raw:: html

    <br>

API (JAX)
^^^^^^^^^

.. autoclass:: skrl.memories.jax.base.Memory
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__
    .. automethod:: __len__
