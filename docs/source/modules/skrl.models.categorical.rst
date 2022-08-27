.. _models_categorical:

Categorical model
=================

Concept
-------

.. image:: ../_static/imgs/model_categorical.svg
    :width: 100%
    :align: center
    :alt: Categorical model

Basic usage
-----------

.. tabs::
    
    .. tab:: Multi-Layer Perceptron (MLP)

        .. literalinclude:: ../snippets/categorical_model.py
            :language: python
            :linenos:
            :start-after: [start-mlp]
            :end-before: [end-mlp]

    .. tab:: Convolutional Neural Network (CNN)

        .. literalinclude:: ../snippets/categorical_model.py
            :language: python
            :linenos:
            :start-after: [start-cnn]
            :end-before: [end-cnn]

API
---

.. autoclass:: skrl.models.torch.categorical.CategoricalMixin
    :show-inheritance:
    :members:

    .. automethod:: __init__
