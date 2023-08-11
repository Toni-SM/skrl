State Action Reward State Action (SARSA)
========================================

SARSA is a **model-free** **on-policy** algorithm that uses a **tabular** Q-function to handle **discrete** observations and action spaces

Paper: `On-Line Q-Learning Using Connectionist Systems <https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.17.2539>`_

.. raw:: html

    <br><hr>

Algorithm
---------

.. raw:: html

    <br>

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - action-value function (:math:`Q`)
|   - states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next states (:math:`s'`), dones (:math:`d`)

.. raw:: html

    <br>

Decision making
"""""""""""""""

|
| :literal:`act(...)`
| :math:`a \leftarrow \pi_{Q[s,a]}(s) \qquad` where :math:`\; a \leftarrow \begin{cases} a \in_R A & x < \epsilon \\ \underset{a}{\arg\max} \; Q[s] & x \geq \epsilon \end{cases} \qquad` for :math:`\; x \leftarrow U(0,1)`

.. raw:: html

    <br>

Learning algorithm
""""""""""""""""""

|
| :literal:`_update(...)`
| :green:`# compute next actions`
| :math:`a' \leftarrow \pi_{Q[s,a]}(s') \qquad` :gray:`# the only difference with Q-learning`
| :green:`# update Q-table`
| :math:`Q[s,a] \leftarrow Q[s,a] \;+` :guilabel:`learning_rate` :math:`(r \;+` :guilabel:`discount_factor` :math:`\neg d \; Q[s',a'] - Q[s,a])`

.. raw:: html

    <br>

Usage
-----

.. tabs::

    .. tab:: Standard implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [torch-start-sarsa]
                    :end-before: [torch-end-sarsa]

.. raw:: html

    <br>

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../skrl/agents/torch/sarsa/sarsa.py
    :language: python
    :start-after: [start-config-dict-torch]
    :end-before: [end-config-dict-torch]

.. raw:: html

    <br>

Spaces
^^^^^^

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

.. raw:: html

    <br>

Models
^^^^^^

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

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.agents.torch.sarsa.SARSA_DEFAULT_CONFIG

.. autoclass:: skrl.agents.torch.sarsa.SARSA
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__
