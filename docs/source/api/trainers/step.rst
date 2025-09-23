Step trainer
============

Train agents controlling the training/evaluation loop step-by-step.

.. raw:: html

    <br><hr>

Concept
-------

.. image:: ../../_static/imgs/manual_trainer-light.svg
    :width: 100%
    :align: center
    :class: only-light
    :alt: Step-by-step trainer

.. image:: ../../_static/imgs/manual_trainer-dark.svg
    :width: 100%
    :align: center
    :class: only-dark
    :alt: Step-by-step trainer

.. raw:: html

    <br>

Usage
-----

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. literalinclude:: ../../snippets/trainer.py
            :language: python
            :start-after: [pytorch-start-step]
            :end-before: [pytorch-end-step]

    .. group-tab:: |_4| |jax| |_4|

        .. literalinclude:: ../../snippets/trainer.py
            :language: python
            :start-after: [jax-start-step]
            :end-before: [jax-end-step]

.. raw:: html

    <br>

Configuration
-------------

.. list-table::
    :header-rows: 1

    * - Dataclass
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - ``StepTrainerCfg``
      - :py:class:`~skrl.trainers.torch.StepTrainerCfg`
      - :py:class:`~skrl.trainers.jax.StepTrainerCfg`

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.trainers.torch.StepTrainerCfg
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.torch.StepTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.trainers.jax.StepTrainerCfg
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.trainers.jax.StepTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
