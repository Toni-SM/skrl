:tocdepth: 4

Trainers
========

.. toctree::
    :hidden:

    Sequential <trainers/sequential>
    Parallel <trainers/parallel>
    Step <trainers/step>

Trainers are responsible for orchestrating and managing the training/evaluation of agents and their interactions with the environment.

.. raw:: html

    <br><hr>

.. list-table::
    :header-rows: 1

    * - Trainers
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`Sequential trainer <trainers/sequential>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Parallel trainer <trainers/parallel>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - :doc:`Step trainer <trainers/step>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

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

        .. literalinclude:: ../snippets/trainer.py
            :language: python
            :start-after: [pytorch-start-base]
            :end-before: [pytorch-end-base]

    .. group-tab:: |_4| |jax| |_4|

        .. literalinclude:: ../snippets/trainer.py
            :language: python
            :start-after: [jax-start-base]
            :end-before: [jax-end-base]

.. raw:: html

    <br>

API (PyTorch)
^^^^^^^^^^^^^

.. autoclass:: skrl.trainers.torch.TrainerCfg
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.torch.Trainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :private-members: _setup_agents
    :members:

    .. automethod:: __str__

.. raw:: html

    <br>

API (JAX)
^^^^^^^^^

.. autoclass:: skrl.trainers.jax.TrainerCfg
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.jax.Trainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :private-members: _setup_agents
    :members:

    .. automethod:: __str__

.. raw:: html

    <br>

API (Warp)
^^^^^^^^^^

.. autoclass:: skrl.trainers.warp.TrainerCfg
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.warp.Trainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :private-members: _setup_agents
    :members:

    .. automethod:: __str__
