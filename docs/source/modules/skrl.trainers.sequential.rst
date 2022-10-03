Sequential trainer
==================

Concept
^^^^^^^

.. image:: ../_static/imgs/sequential_trainer.svg
    :width: 100%
    :align: center
    :alt: Sequential trainer

Basic usage
^^^^^^^^^^^

.. tabs::
            
    .. tab:: Snippet

        .. literalinclude:: ../snippets/trainer.py
            :language: python
            :linenos:
            :start-after: [start-sequential]
            :end-before: [end-sequential]

Configuration
^^^^^^^^^^^^^

.. py:data:: skrl.trainers.torch.sequential.SEQUENTIAL_TRAINER_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/trainers/torch/sequential.py
    :language: python
    :lines: 14-17
    :linenos:

API
^^^

.. autoclass:: skrl.trainers.torch.sequential.SequentialTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__
