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

    .. tab:: Snippet

        .. literalinclude:: ../../snippets/trainer.py
            :language: python
            :linenos:
            :start-after: [start-sequential]
            :end-before: [end-sequential]

.. raw:: html

    <br>

Configuration
-------------

.. literalinclude:: ../../../../skrl/trainers/torch/sequential.py
    :language: python
    :lines: 14-18
    :linenos:

.. raw:: html

    <br>

API
---

.. autoclass:: skrl.trainers.torch.sequential.SEQUENTIAL_TRAINER_DEFAULT_CONFIG

.. autoclass:: skrl.trainers.torch.sequential.SequentialTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__
