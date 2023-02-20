State Action Reward State Action (SARSA)
========================================

SARSA is a **model-free** **on-policy** algorithm that uses a **tabular** Q-function to handle **discrete** observations and action spaces

Paper: `On-Line Q-Learning Using Connectionist Systems <https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.2539>`_

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^
| Main notation/symbols:
|   - action-value function (:math:`Q`)
|   - states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next states (:math:`s'`), dones (:math:`d`)

**Decision making** (:literal:`act(...)`)

| :math:`a \leftarrow \pi_{Q[s,a]}(s) \qquad` where :math:`\; a \leftarrow \begin{cases} a \in_R A & x < \epsilon \\ \underset{a}{\arg\max} \; Q[s] & x \geq \epsilon \end{cases} \qquad` for :math:`\; x \leftarrow U(0,1)`

**Learning algorithm** (:literal:`_update(...)`)

| :green:`# compute next actions`
| :math:`a' \leftarrow \pi_{Q[s,a]}(s') \qquad` :gray:`# the only difference with Q-learning`
| :green:`# update Q-table`
| :math:`Q[s,a] \leftarrow Q[s,a] \;+` :guilabel:`learning_rate` :math:`(r \;+` :guilabel:`discount_factor` :math:`\neg d \; Q[s',a'] - Q[s,a])`

Basic usage
^^^^^^^^^^^

.. tabs::

    .. tab:: Standard implementation

        .. literalinclude:: ../../snippets/agents_basic_usage.py
            :language: python
            :emphasize-lines: 2
            :start-after: [start-sarsa]
            :end-before: [end-sarsa]

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: skrl.agents.torch.sarsa.sarsa.SARSA_DEFAULT_CONFIG

.. literalinclude:: ../../../../skrl/agents/torch/sarsa/sarsa.py
    :language: python
    :lines: 14-35
    :linenos:

Spaces and models
^^^^^^^^^^^^^^^^^

The implementation supports the following `Gym spaces <https://www.gymlibrary.dev/api/spaces>`_ / `Gymnasium spaces <https://gymnasium.farama.org/api/spaces>`_

.. list-table::
    :header-rows: 1

    * - Gym/Gymnasium spaces
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
      - Input shape
      - Output shape
      - Type
    * - :math:`\pi_{Q[s,a]}(s)`
      - Policy (:math:`\epsilon`-greedy)
      - :literal:`"policy"`
      - observation
      - action
      - :ref:`Tabular <models_tabular>`

API
^^^

.. autoclass:: skrl.agents.torch.sarsa.sarsa.SARSA
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__