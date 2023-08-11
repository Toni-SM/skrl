Deep Deterministic Policy Gradient (DDPG)
=========================================

DDPG is a **model-free**, **deterministic** **off-policy** **actor-critic** algorithm that uses deep function approximators to learn a policy (and to estimate the action-value function) in high-dimensional, **continuous** action spaces

Paper: `Continuous control with deep reinforcement learning <https://arxiv.org/abs/1509.02971>`_

.. raw:: html

    <br><hr>

Algorithm
---------

.. raw:: html

    <br>

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - policy function approximator (:math:`\mu_\theta`), critic function approximator (:math:`Q_\phi`)
|   - states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next states (:math:`s'`), dones (:math:`d`)
|   - loss (:math:`L`)

.. raw:: html

    <br>

Decision making
"""""""""""""""

|
| :literal:`act(...)`
| :math:`a \leftarrow \mu_\theta(s)`
| :math:`noise \leftarrow` sample :guilabel:`noise`
| :math:`scale \leftarrow (1 - \text{timestep} \;/` :guilabel:`timesteps` :math:`) \; (` :guilabel:`initial_scale` :math:`-` :guilabel:`final_scale` :math:`) \;+` :guilabel:`final_scale`
| :math:`a \leftarrow \text{clip}(a + noise * scale, {a}_{Low}, {a}_{High})`

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
|     :math:`a' \leftarrow \mu_{\theta_{target}}(s')`
|     :math:`Q_{_{target}} \leftarrow Q_{\phi_{target}}(s', a')`
|     :math:`y \leftarrow r \;+` :guilabel:`discount_factor` :math:`\neg d \; Q_{_{target}}`
|     :green:`# compute critic loss`
|     :math:`Q \leftarrow Q_\phi(s, a)`
|     :math:`L_{Q_\phi} \leftarrow \frac{1}{N} \sum_{i=1}^N (Q - y)^2`
|     :green:`# optimization step (critic)`
|     reset :math:`\text{optimizer}_\phi`
|     :math:`\nabla_{\phi} L_{Q_\phi}`
|     :math:`\text{clip}(\lVert \nabla_{\phi} \rVert)` with :guilabel:`grad_norm_clip`
|     step :math:`\text{optimizer}_\phi`
|     :green:`# compute policy (actor) loss`
|     :math:`a \leftarrow \mu_\theta(s)`
|     :math:`Q \leftarrow Q_\phi(s, a)`
|     :math:`L_{\mu_\theta} \leftarrow - \frac{1}{N} \sum_{i=1}^N Q`
|     :green:`# optimization step (policy)`
|     reset :math:`\text{optimizer}_\theta`
|     :math:`\nabla_{\theta} L_{\mu_\theta}`
|     :math:`\text{clip}(\lVert \nabla_{\theta} \rVert)` with :guilabel:`grad_norm_clip`
|     step :math:`\text{optimizer}_\theta`
|     :green:`# update target networks`
|     :math:`\theta_{target} \leftarrow` :guilabel:`polyak` :math:`\theta + (1 \;-` :guilabel:`polyak` :math:`) \theta_{target}`
|     :math:`\phi_{target} \leftarrow` :guilabel:`polyak` :math:`\phi + (1 \;-` :guilabel:`polyak` :math:`) \phi_{target}`
|     :green:`# update learning rate`
|     **IF** there is a :guilabel:`learning_rate_scheduler` **THEN**
|         step :math:`\text{scheduler}_\theta (\text{optimizer}_\theta)`
|         step :math:`\text{scheduler}_\phi (\text{optimizer}_\phi)`

.. raw:: html

    <br>

Usage
-----

.. note::

    Support for recurrent neural networks (RNN, LSTM, GRU and any other variant) is implemented in a separate file (:literal:`ddpg_rnn.py`) to maintain the readability of the standard implementation (:literal:`ddpg.py`)

.. tabs::

    .. tab:: Standard implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [torch-start-ddpg]
                    :end-before: [torch-end-ddpg]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [jax-start-ddpg]
                    :end-before: [jax-end-ddpg]

    .. tab:: RNN implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. note::

                    When using recursive models it is necessary to override their :literal:`.get_specification()` method. Visit each model's documentation for more details

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [torch-start-ddpg-rnn]
                    :end-before: [torch-end-ddpg-rnn]

.. raw:: html

    <br>

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../skrl/agents/torch/ddpg/ddpg.py
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
      - .. centered:: :math:`\square`
    * - Box
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - Dict
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

.. raw:: html

    <br>

Models
^^^^^^

The implementation uses 4 deterministic function approximators. These function approximators (models) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`models`

.. list-table::
    :header-rows: 1

    * - Notation
      - Concept
      - Key
      - Input shape
      - Output shape
      - Type
    * - :math:`\mu_\theta(s)`
      - Policy (actor)
      - :literal:`"policy"`
      - observation
      - action
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`\mu_{\theta_{target}}(s)`
      - Target policy
      - :literal:`"target_policy"`
      - observation
      - action
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_\phi(s, a)`
      - Q-network (critic)
      - :literal:`"critic"`
      - observation + action
      - 1
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_{\phi_{target}}(s, a)`
      - Target Q-network
      - :literal:`"target_critic"`
      - observation + action
      - 1
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
      - RNN, LSTM, GRU and any other variant
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.agents.torch.ddpg.DDPG_DEFAULT_CONFIG

.. autoclass:: skrl.agents.torch.ddpg.DDPG
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.agents.torch.ddpg.DDPG_RNN
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.agents.jax.ddpg.DDPG_DEFAULT_CONFIG

.. autoclass:: skrl.agents.jax.ddpg.DDPG
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__
