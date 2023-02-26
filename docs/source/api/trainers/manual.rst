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

    .. tab:: Snippet

        .. literalinclude:: ../../snippets/trainer.py
            :language: python
            :linenos:
            :start-after: [start-manual]
            :end-before: [end-manual]

.. raw:: html

    <br>

Configuration
-------------

.. literalinclude:: ../../../../skrl/trainers/torch/manual.py
    :language: python
    :lines: 14-18
    :linenos:

.. raw:: html

    <br>

API
---

.. autoclass:: skrl.trainers.torch.manual.MANUAL_TRAINER_DEFAULT_CONFIG

.. autoclass:: skrl.trainers.torch.manual.ManualTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__
