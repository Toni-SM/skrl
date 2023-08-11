Double Deep Q-Network (DDQN)
============================

DDQN is a **model-free**, **off-policy** algorithm that relies on double Q-learning to avoid the overestimation of action-values introduced by DQN

Paper: `Deep Reinforcement Learning with Double Q-Learning <https://ojs.aaai.org/index.php/AAAI/article/view/10295>`_

.. raw:: html

    <br><hr>

Algorithm
---------

.. raw:: html

    <br>

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

    <br>

Decision making
"""""""""""""""

|
| :literal:`act(...)`
| :math:`\epsilon \leftarrow \epsilon_{_{final}} + (\epsilon_{_{initial}} - \epsilon_{_{final}}) \; e^{-1 \; \frac{\text{timestep}}{\epsilon_{_{timesteps}}}}`
| :math:`a \leftarrow \begin{cases} a \in_R A & x < \epsilon \\ \underset{a}{\arg\max} \; Q_\phi(s) & x \geq \epsilon \end{cases} \qquad` for :math:`\; x \leftarrow U(0,1)`

.. raw:: html

    <br>

Learning algorithm
""""""""""""""""""

|
| :literal:`_update(...)`
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
                    :start-after: [torch-start-ddqn]
                    :end-before: [torch-end-ddqn]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [jax-start-ddqn]
                    :end-before: [jax-end-ddqn]

.. raw:: html

    <br>

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../skrl/agents/torch/dqn/ddqn.py
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
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\blacksquare`
    * - Box
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - Dict
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

.. raw:: html

    <br>

Models
^^^^^^

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

.. raw:: html

    <br>

Features
^^^^^^^^

Support for advanced features is described in the next table

.. list-table::
    :header-rows: 1

    * - Feature
      - Support and remarks
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - Shared model
      - \-
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - RNN support
      - \-
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.agents.torch.dqn.DDQN_DEFAULT_CONFIG

.. autoclass:: skrl.agents.torch.dqn.DDQN
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.agents.jax.dqn.DDQN_DEFAULT_CONFIG

.. autoclass:: skrl.agents.jax.dqn.DDQN
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__
