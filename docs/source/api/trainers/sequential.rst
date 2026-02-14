:tocdepth: 4

Sequential trainer
==================

Train agents sequentially (i.e., one after the other in each interaction with the environment).

|br| |hr|

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

|

Configuration
-------------

.. list-table::
    :header-rows: 1

    * - Dataclass
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - ``SequentialTrainerCfg``
      - :py:class:`~skrl.trainers.torch.sequential.SequentialTrainerCfg`
      - :py:class:`~skrl.trainers.jax.sequential.SequentialTrainerCfg`
      - :py:class:`~skrl.trainers.warp.sequential.SequentialTrainerCfg`

|

API
---

|

PyTorch
^^^^^^^

.. automodule:: skrl.trainers.torch.sequential
.. autosummary::
    :nosignatures:

    SequentialTrainerCfg
    SequentialTrainer

.. autoclass:: skrl.trainers.torch.sequential.SequentialTrainerCfg
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.torch.sequential.SequentialTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

JAX
^^^

.. automodule:: skrl.trainers.jax.sequential
.. autosummary::
    :nosignatures:

    SequentialTrainerCfg
    SequentialTrainer

.. autoclass:: skrl.trainers.jax.sequential.SequentialTrainerCfg
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.jax.sequential.SequentialTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

Warp
^^^^

.. automodule:: skrl.trainers.warp.sequential
.. autosummary::
    :nosignatures:

    SequentialTrainerCfg
    SequentialTrainer

.. autoclass:: skrl.trainers.warp.sequential.SequentialTrainerCfg
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.warp.sequential.SequentialTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
