Sequential trainer
==================

Concept
^^^^^^^

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

Basic usage
^^^^^^^^^^^

.. tabs::

    .. tab:: Snippet

        .. literalinclude:: ../../snippets/trainer.py
            :language: python
            :linenos:
            :start-after: [start-sequential]
            :end-before: [end-sequential]

Configuration
^^^^^^^^^^^^^

.. py:data:: skrl.trainers.torch.sequential.SEQUENTIAL_TRAINER_DEFAULT_CONFIG

.. literalinclude:: ../../../../skrl/trainers/torch/sequential.py
    :language: python
    :lines: 14-18
    :linenos:

API
^^^

.. autoclass:: skrl.trainers.torch.sequential.SequentialTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__
