Q-learning
==========

Q-learning is a **model-free** **off-policy** algorithm that uses a **tabular** Q-function to handle **discrete** observations and action spaces

Paper: `Learning from delayed rewards <https://www.academia.edu/3294050/Learning_from_delayed_rewards>`_

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - action-value function (:math:`Q`)
|   - states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next states (:math:`s'`), dones (:math:`d`)

**Decision making** (:literal:`act(...)`)

| :math:`a \leftarrow \pi_{Q[s,a]}(s) \qquad` where :math:`\; a \leftarrow \begin{cases} a \in_R A & x < \epsilon \\ \underset{a}{\arg\max} \; Q[s] & x \geq \epsilon \end{cases} \qquad` for :math:`\; x \leftarrow U(0,1)`

**Learning algorithm** (:literal:`_update(...)`)

| :green:`# compute next actions`
| :math:`a' \leftarrow \underset{a}{\arg\max} \; Q[s'] \qquad` :gray:`# the only difference with SARSA`
| :green:`# update Q-table`
| :math:`Q[s,a] \leftarrow Q[s,a] \;+` :guilabel:`learning_rate` :math:`(r \;+` :guilabel:`discount_factor` :math:`\neg d \; Q[s',a'] - Q[s,a])`

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: skrl.agents.torch.q_learning.q_learning.Q_LEARNING_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/q_learning/q_learning.py
   :language: python
   :lines: 14-32
   :linenos:

Spaces and models
^^^^^^^^^^^^^^^^^

The implementation supports the following `Gym spaces <https://www.gymlibrary.ml/content/spaces>`_

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
   * - Dict
     - .. centered:: :math:`\square`
     - .. centered:: :math:`\square`

The implementation uses 1 table. This table (model) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`models`

.. list-table::
   :header-rows: 1

   * - Notation
     - Concept
     - Key
     - Type
   * - :math:`\pi_{Q[s,a]}(s)`
     - Policy (:math:`\epsilon`-greedy)
     - :literal:`"policy"`
     - :ref:`Tabular <models_tabular>`

API
^^^

.. autoclass:: skrl.agents.torch.q_learning.q_learning.Q_LEARNING
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
