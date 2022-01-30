.. _models_deterministic:

Deterministic model
===================

Concept
^^^^^^^

.. image:: ../_static/imgs/model_deterministic.png
      :width: 75%
      :align: center
      :alt: Deterministic model

Basic usage
^^^^^^^^^^^

View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/snippets/deterministic_model.py>`_

.. literalinclude:: ../snippets/deterministic_model.py
    :language: python
    :linenos:

API
^^^

.. autoclass:: skrl.models.torch.deterministic.DeterministicModel
   :undoc-members:
   :show-inheritance:
   :members:
   
   .. automethod:: __init__
   .. automethod:: compute
