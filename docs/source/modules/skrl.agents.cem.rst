Cross-Entropy Method (CEM)
==========================

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

**Decision making** (:literal:`act(...)`)

| :math:`a \leftarrow \pi_\theta(s)`

**Learning algorithm** (:literal:`_update(...)`)

| :green:`# sample all memory`
| :math:`s, a, r, s', d \leftarrow` states, actions, rewards, next_states, dones
| :green:`# compute discounted return threshold`
| :math:`[G] \leftarrow \sum_{t=0}^{E-1} \gamma^{t} r_t` for each episode
| :math:`G_{_{bound}} \leftarrow q_{th_{percentile}}([G])`
| :green:`# get elite states and actions`
| :math:`s_{_{elite}} \leftarrow s[G \geq G_{_{bound}}]`
| :math:`a_{_{elite}} \leftarrow a[G \geq G_{_{bound}}]`
| :green:`# compute scores for the elite states`
| :math:`scores \leftarrow \theta(s_{_{elite}})`
| :green:`# compute policy loss`
| :math:`{Loss}_{policy} \leftarrow -\sum_{i=1}^{N} a_{_{elite}} \log(scores)`
| :green:`# optimize policy`
| :math:`\nabla_{\theta} {Loss}_{policy}`

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: skrl.agents.torch.cem.cem.CEM_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/cem/cem.py
   :language: python
   :lines: 15-41
   :linenos:

Spaces and models
^^^^^^^^^^^^^^^^^

The implementation supports the following `Gym spaces <https://gym.openai.com/docs/#spaces>`_

.. list-table::
   :header-rows: 1

   * - Gym spaces
     - .. centered:: Observation
     - .. centered:: Action
   * - Discrete
     - .. centered:: :math:`\square`
     - .. centered:: :math:`\blacksquare`
   * - Box
     - .. centered:: :math:`\blacksquare`
     - .. centered:: :math:`\square`
   * - Dict
     - .. centered:: :math:`\blacksquare`
     - .. centered:: :math:`\square`

The implementation uses 1 discrete function approximator. This function approximator (model) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`models`

.. list-table::
   :header-rows: 1

   * - Notation
     - Concept
     - Key
     - Type
   * - :math:`\pi(s)`
     - Policy
     - :literal:`"policy"`
     - :ref:`Categorical <models_categorical>`

API
^^^

.. autoclass:: skrl.agents.torch.cem.cem.CEM
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
