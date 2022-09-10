.. _models_tabular:

Tabular model
=============

Basic usage
-----------

.. tabs::
    
    .. tab:: :math:`\epsilon`-greedy policy

        .. literalinclude:: ../snippets/tabular_model.py
            :language: python
            :linenos:
            :start-after: [start-epsilon-greedy]
            :end-before: [end-epsilon-greedy]

API
---

.. autoclass:: skrl.models.torch.tabular.TabularMixin
    :show-inheritance:
    :exclude-members: to, state_dict, load_state_dict, load, save
    :members:

    .. automethod:: __init__
