Manual trainer
==============

Train agents by manually controlling the training/evaluation loop.

.. raw:: html

    <br><hr>

Concept
-------

.. image:: ../../_static/imgs/manual_trainer-light.svg
    :width: 100%
    :align: center
    :class: only-light
    :alt: Manual trainer

.. image:: ../../_static/imgs/manual_trainer-dark.svg
    :width: 100%
    :align: center
    :class: only-dark
    :alt: Manual trainer

.. raw:: html

    <br>

Usage
-----

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. literalinclude:: ../../snippets/trainer.py
            :language: python
            :start-after: [pytorch-start-manual]
            :end-before: [pytorch-end-manual]

    .. group-tab:: |_4| |jax| |_4|

        .. literalinclude:: ../../snippets/trainer.py
            :language: python
            :start-after: [jax-start-manual]
            :end-before: [jax-end-manual]

.. raw:: html

    <br>

Configuration
-------------

.. literalinclude:: ../../../../skrl/trainers/torch/manual.py
    :language: python
    :lines: 14-19
    :linenos:

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.trainers.torch.manual.MANUAL_TRAINER_DEFAULT_CONFIG

.. autoclass:: skrl.trainers.torch.manual.ManualTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.trainers.jax.manual.MANUAL_TRAINER_DEFAULT_CONFIG

.. autoclass:: skrl.trainers.jax.manual.ManualTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__
