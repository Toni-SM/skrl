.. _models_gaussian:

Gaussian model
==============

Concept
^^^^^^^

.. image:: ../_static/imgs/model_gaussian.png
      :width: 100%
      :align: center
      :alt: Gaussian model

Basic usage
^^^^^^^^^^^

.. tabs::
    
    .. tab:: Multi-Layer Perceptron (MLP)

        .. literalinclude:: ../snippets/gaussian_model.py
            :language: python
            :linenos:
            :start-after: [start-mlp]
            :end-before: [end-mlp]

    .. tab:: Convolutional Neural Network (CNN)

        .. literalinclude:: ../snippets/gaussian_model.py
            :language: python
            :linenos:
            :start-after: [start-cnn]
            :end-before: [end-cnn]

API
^^^

.. autoclass:: skrl.models.torch.gaussian.GaussianModel
   :undoc-members:
   :show-inheritance:
   :members:
   
   .. automethod:: __init__
   .. automethod:: compute
