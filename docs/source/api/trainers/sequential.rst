:tocdepth: 3

Sequential trainer
==================

Train agents sequentially (i.e., one after the other in each interaction with the environment).

.. raw:: html

    <br><hr>

Concept
-------

.. image:: ../../_static/imgs/sequential_trainer-light.svg
    :width: 100%
    :align: center
    :class: only-light
    :alt: Sequential trainer

.. image:: ../../_static/imgs/sequential_trainer-dark.svg
    :width: 100%
    :align: center
    :class: only-dark
    :alt: Sequential trainer

.. raw:: html

    <br>

Usage
-----

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. literalinclude:: ../../snippets/trainer.py
            :language: python
            :start-after: [pytorch-start-sequential]
            :end-before: [pytorch-end-sequential]

    .. group-tab:: |_4| |jax| |_4|

        .. literalinclude:: ../../snippets/trainer.py
            :language: python
            :start-after: [jax-start-sequential]
            :end-before: [jax-end-sequential]

    .. group-tab:: |_4| |warp| |_4|

        .. literalinclude:: ../../snippets/trainer.py
            :language: python
            :start-after: [warp-start-sequential]
            :end-before: [warp-end-sequential]

.. raw:: html

    <br>

Configuration
-------------

.. list-table::
    :header-rows: 1

    * - Dataclass
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - ``SequentialTrainerCfg``
      - :py:class:`~skrl.trainers.torch.SequentialTrainerCfg`
      - :py:class:`~skrl.trainers.jax.SequentialTrainerCfg`
      - :py:class:`~skrl.trainers.warp.SequentialTrainerCfg`

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.trainers.torch.SequentialTrainerCfg
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.torch.SequentialTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.trainers.jax.SequentialTrainerCfg
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.jax.SequentialTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. raw:: html

    <br>

API (Warp)
----------

.. autoclass:: skrl.trainers.warp.SequentialTrainerCfg
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.warp.SequentialTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
