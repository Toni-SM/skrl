Base class
==========

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.

Basic inheritance usage
^^^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. tab:: Inheritance

        .. literalinclude:: ../snippets/trainer.py
            :language: python
            :linenos:
            :start-after: [start-base]
            :end-before: [end-base]

API
^^^

.. autoclass:: skrl.trainers.torch.base.Trainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :private-members: _setup_agents
    :members:

    .. automethod:: __init__
    .. automethod:: __str__
