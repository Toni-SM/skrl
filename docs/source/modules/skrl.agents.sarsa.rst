State Action Reward State Action (SARSA)
========================================

SARSA is a **model-free** **on-policy** algorithm that uses a **tabular** Q-function to handle **discrete** observations and action spaces

Paper: `On-Line Q-Learning Using Connectionist Systems <https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.2539>`_

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

**Decision making** (:literal:`act(...)`)

| :math:`a \leftarrow \underset{a}{\arg\max} \; Q[s]`

**Learning algorithm** (:literal:`_update(...)`)

| :green:`# compute next actions`
| :math:`a' \leftarrow \underset{a}{\arg\max} \; Q[s']`
| :green:`# update Q-table`
| :math:`Q[s,a] \leftarrow Q[s,a] + \alpha \; (r + \gamma \; \neg d \; Q[s',a'] - Q[s,a])`

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: skrl.agents.torch.sarsa.sarsa.SARSA_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/sarsa/sarsa.py
   :language: python
   :lines: 14-31
   :linenos:

Spaces and models
^^^^^^^^^^^^^^^^^

The implementation supports the following `Gym spaces <https://gym.openai.com/docs/#spaces>`_:

.. list-table::
   :header-rows: 1

   * - Gym spaces
     - .. centered:: Observation
     - .. centered:: Action
   * - Discrete
     - .. centered:: :math:`\blacksquare`
     - .. centered:: :math:`\blacksquare`
   * - Box
     - .. centered:: :math:`\square`
     - .. centered:: :math:`\square`

The implementation uses 1 table. This table (model) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`models`

.. list-table::
   :header-rows: 1

   * - Notation
     - Concept
     - Key
     - Type
   * - :math:`Q[s,a]`
     - Q-table
     - :literal:`"policy"`
     - :ref:`Tabular <models_tabular>`

API
^^^

.. autoclass:: skrl.agents.torch.sarsa.sarsa.SARSA
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
