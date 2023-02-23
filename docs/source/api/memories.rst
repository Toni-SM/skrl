Memories
========

.. toctree::
    :hidden:

    Random <memories/random>

Base class
----------

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.

Basic inheritance usage
^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. tab:: Inheritance

        .. literalinclude:: ../snippets/memory.py
            :language: python
            :linenos:

API
^^^

.. autoclass:: skrl.memories.torch.base.Memory
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__
    .. automethod:: __len__
