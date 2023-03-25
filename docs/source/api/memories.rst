Memories
========

.. toctree::
    :hidden:

    Random <memories/random>

Memories are storage components that allow agents to collect and use/reuse current or past experiences of their interaction with the environment or other types of information.

.. raw:: html

    <br><hr>

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

    .. tab:: Inheritance

        .. literalinclude:: ../snippets/memory.py
            :language: python
            :linenos:

.. raw:: html

    <br>

API
^^^

.. autoclass:: skrl.memories.torch.base.Memory
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__
    .. automethod:: __len__
