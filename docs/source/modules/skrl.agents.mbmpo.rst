Model-Based Meta-Policy-Optimization (MBMPO)
============================================

MBMPO is a **model-based** algorithm that uses an ensemble of learned dynamic models to meta-learn a policy that can quickly adapt to any model in the ensemble with one policy gradient step.

Paper: `Model-Based Reinforcement Learning via Meta-Policy Optimization <https://arxiv.org/abs/1809.05214>`_

Algorithm
^^^^^^^^^

TODO :red:`(comming soon)`

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

**Learning algorithm** (:literal:`_update(...)`)

TODO :red:`(comming soon)`

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: skrl.agents.torch.trpo.trpo.TRPO_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/trpo/trpo.py
   :language: python
   :lines: 18-
   :linenos:

Spaces and models
^^^^^^^^^^^^^^^^^

The implementation supports the following `Gym spaces <https://www.gymlibrary.dev/content/spaces>`_

TODO :red:`(comming soon)`

API
^^^

.. autoclass:: skrl.agents.torch.mbmpo.mbmpo.MBMPO
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:

   .. automethod:: __init__
