Double Deep Q-Network (DDQN)
============================

DDQN is a **model-free**, **off-policy** algorithm that relies on double Q-learning to avoid the overestimation of action-values introduced by DQN

Paper: `Deep Reinforcement Learning with Double Q-Learning <https://ojs.aaai.org/index.php/AAAI/article/view/10295>`_

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

**Decision making** (:literal:`act(...)`)

| :math:`\epsilon \leftarrow \epsilon_{_{final}} + (\epsilon_{_{initial}} - \epsilon_{_{final}}) \; e^{-1 \; \frac{\text{timestep}}{\epsilon_{_{timesteps}}}}`
| :math:`a \leftarrow \begin{cases} a \in_R A & x < \epsilon \\ \underset{a}{\arg\max} \; Q_\phi(s) & x \geq \epsilon \end{cases} \qquad` for :math:`\; x \leftarrow U(0,1)`

**Learning algorithm** (:literal:`_update(...)`)

| :green:`# sample a batch from memory`
| [:math:`s, a, r, s', d`] :math:`\leftarrow` states, actions, rewards, next_states, dones of size :guilabel:`batch_size`
| :green:`# gradient steps`
| **FOR** each gradient step up to :guilabel:`gradient_steps` **DO**
|     :green:`# compute target values`
|     :math:`Q' \leftarrow Q_{\phi_{target}}(s')`
|     :math:`Q_{_{target}} \leftarrow Q'[\underset{a}{\arg\max} \; Q_\phi(s')] \qquad` :gray:`# the only difference with DQN`
|     :math:`y \leftarrow r \;+` :guilabel:`discount_factor` :math:`\neg d \; Q_{_{target}}`
|     :green:`# compute Q-network loss`
|     :math:`Q \leftarrow Q_\phi(s)[a]`
|     :math:`{Loss}_{Q_\phi} \leftarrow \frac{1}{N} \sum_{i=1}^N (Q - y)^2`
|     :green:`# optimize Q-network`
|     :math:`\nabla_{\phi} {Loss}_{Q_\phi}`
|     :green:`# update target network`
|     **IF** it's time to update target network **THEN**
|         :math:`\phi_{target} \leftarrow` :guilabel:`polyak` :math:`\phi + (1 \;-` :guilabel:`polyak` :math:`) \phi_{target}`
|     :green:`# update learning rate`
|     **IF** there is a :guilabel:`learning_rate_scheduler` **THEN**
|         step :math:`\text{scheduler}_\phi (\text{optimizer}_\phi)`

Basic usage
^^^^^^^^^^^

.. tabs::

    .. tab:: Standard implementation

        .. literalinclude:: ../snippets/agents_basic_usage.py
            :language: python
            :emphasize-lines: 2
            :start-after: [start-ddqn]
            :end-before: [end-ddqn]

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: skrl.agents.torch.dqn.ddqn.DDQN_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/dqn/ddqn.py
    :language: python
    :lines: 16-55
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
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\blacksquare`
    * - Box
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - Dict
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

The implementation uses 2 deterministic function approximators. These function approximators (models) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`models`

.. list-table::
    :header-rows: 1

    * - Notation
      - Concept
      - Key
      - Input shape
      - Output shape
      - Type
    * - :math:`Q_\phi(s, a)`
      - Q-network
      - :literal:`"q_network"`
      - observation
      - action
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_{\phi_{target}}(s, a)`
      - Target Q-network
      - :literal:`"target_q_network"`
      - observation
      - action
      - :ref:`Deterministic <models_deterministic>`

Support for advanced features is described in the next table

.. list-table::
    :header-rows: 1

    * - Feature
      - Support and remarks
    * - Shared model
      - \-
    * - RNN support
      - \-

API
^^^

.. autoclass:: skrl.agents.torch.dqn.ddqn.DDQN
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__
