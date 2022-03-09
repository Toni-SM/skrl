.. _models_deterministic:

Deterministic model
===================

Concept
^^^^^^^

.. image:: ../_static/imgs/model_deterministic.svg
      :width: 65%
      :align: center
      :alt: Deterministic model

Basic usage
^^^^^^^^^^^

.. tabs::
    
    .. tab:: Multi-Layer Perceptron (MLP)

        .. literalinclude:: ../snippets/deterministic_model.py
            :language: python
            :linenos:
            :start-after: [start-mlp]
            :end-before: [end-mlp]

    .. tab:: Convolutional Neural Network (CNN)

        .. literalinclude:: ../snippets/deterministic_model.py
            :language: python
            :linenos:
            :start-after: [start-cnn]
            :end-before: [end-cnn]

API
^^^

.. autoclass:: skrl.models.torch.deterministic.DeterministicModel
   :show-inheritance:
   :members:
   
   .. automethod:: __init__
   .. automethod:: compute
