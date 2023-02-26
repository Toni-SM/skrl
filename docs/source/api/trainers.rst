Trainers
========

.. toctree::
    :hidden:

    Sequential <trainers/sequential>
    Parallel <trainers/parallel>
    Manual <trainers/manual>

Trainers are responsible for orchestrating and managing the training/evaluation of agents and their interactions with the environment.

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

        .. literalinclude:: ../snippets/trainer.py
            :language: python
            :linenos:
            :start-after: [start-base]
            :end-before: [end-base]

.. raw:: html

    <br>

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
