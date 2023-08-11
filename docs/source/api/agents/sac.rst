Soft Actor-Critic (SAC)
=======================

SAC is a **model-free**, **stochastic** **off-policy** **actor-critic** algorithm that uses double Q-learning (like TD3) and **entropy** regularization to maximize a trade-off between exploration and exploitation

Paper: `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_

.. raw:: html

    <br><hr>

Algorithm
---------

.. raw:: html

    <br>

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - policy function approximator (:math:`\pi_\theta`), critic function approximator (:math:`Q_\phi`)
|   - states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next states (:math:`s'`), dones (:math:`d`)
|   - log probabilities (:math:`logp`), entropy coefficient (:math:`\alpha`)
|   - loss (:math:`L`)

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
|     :math:`a',\; logp' \leftarrow \pi_\theta(s')`
|     :math:`Q_{1_{target}} \leftarrow Q_{{\phi 1}_{target}}(s', a')`
|     :math:`Q_{2_{target}} \leftarrow Q_{{\phi 2}_{target}}(s', a')`
|     :math:`Q_{_{target}} \leftarrow \text{min}(Q_{1_{target}}, Q_{2_{target}}) - \alpha \; logp'`
|     :math:`y \leftarrow r \;+` :guilabel:`discount_factor` :math:`\neg d \; Q_{_{target}}`
|     :green:`# compute critic loss`
|     :math:`Q_1 \leftarrow Q_{\phi 1}(s, a)`
|     :math:`Q_2 \leftarrow Q_{\phi 2}(s, a)`
|     :math:`L_{Q_\phi} \leftarrow 0.5 \; (\frac{1}{N} \sum_{i=1}^N (Q_1 - y)^2 + \frac{1}{N} \sum_{i=1}^N (Q_2 - y)^2)`
|     :green:`# optimization step (critic)`
|     reset :math:`\text{optimizer}_\phi`
|     :math:`\nabla_{\phi} L_{Q_\phi}`
|     :math:`\text{clip}(\lVert \nabla_{\phi} \rVert)` with :guilabel:`grad_norm_clip`
|     step :math:`\text{optimizer}_\phi`
|     :green:`# compute policy (actor) loss`
|     :math:`a,\; logp \leftarrow \pi_\theta(s)`
|     :math:`Q_1 \leftarrow Q_{\phi 1}(s, a)`
|     :math:`Q_2 \leftarrow Q_{\phi 2}(s, a)`
|     :math:`L_{\pi_\theta} \leftarrow \frac{1}{N} \sum_{i=1}^N (\alpha \; logp - \text{min}(Q_1, Q_2))`
|     :green:`# optimization step (policy)`
|     reset :math:`\text{optimizer}_\theta`
|     :math:`\nabla_{\theta} L_{\pi_\theta}`
|     :math:`\text{clip}(\lVert \nabla_{\theta} \rVert)` with :guilabel:`grad_norm_clip`
|     step :math:`\text{optimizer}_\theta`
|     :green:`# entropy learning`
|     **IF** :guilabel:`learn_entropy` is enabled **THEN**
|         :green:`# compute entropy loss`
|         :math:`{L}_{entropy} \leftarrow - \frac{1}{N} \sum_{i=1}^N (log(\alpha) \; (logp + \alpha_{Target}))`
|         :green:`# optimization step (entropy)`
|         reset :math:`\text{optimizer}_\alpha`
|         :math:`\nabla_{\alpha} {L}_{entropy}`
|         step :math:`\text{optimizer}_\alpha`
|         :green:`# compute entropy coefficient`
|         :math:`\alpha \leftarrow e^{log(\alpha)}`
|     :green:`# update target networks`
|     :math:`{\phi 1}_{target} \leftarrow` :guilabel:`polyak` :math:`{\phi 1} + (1 \;-` :guilabel:`polyak` :math:`) {\phi 1}_{target}`
|     :math:`{\phi 2}_{target} \leftarrow` :guilabel:`polyak` :math:`{\phi 2} + (1 \;-` :guilabel:`polyak` :math:`) {\phi 2}_{target}`
|     :green:`# update learning rate`
|     **IF** there is a :guilabel:`learning_rate_scheduler` **THEN**
|         step :math:`\text{scheduler}_\theta (\text{optimizer}_\theta)`
|         step :math:`\text{scheduler}_\phi (\text{optimizer}_\phi)`

.. raw:: html

    <br>

Usage
-----

.. note::

    Support for recurrent neural networks (RNN, LSTM, GRU and any other variant) is implemented in a separate file (:literal:`sac_rnn.py`) to maintain the readability of the standard implementation (:literal:`sac.py`)

.. tabs::

    .. tab:: Standard implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [torch-start-sac]
                    :end-before: [torch-end-sac]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [jax-start-sac]
                    :end-before: [jax-end-sac]

    .. tab:: RNN implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. note::

                    When using recursive models it is necessary to override their :literal:`.get_specification()` method. Visit each model's documentation for more details

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [torch-start-sac-rnn]
                    :end-before: [torch-end-sac-rnn]

.. raw:: html

    <br>

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../skrl/agents/torch/sac/sac.py
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

The implementation uses 1 stochastic and 4 deterministic function approximators. These function approximators (models) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`models`

.. list-table::
    :header-rows: 1

    * - Notation
      - Concept
      - Key
      - Input shape
      - Output shape
      - Type
    * - :math:`\pi_\theta(s)`
      - Policy (actor)
      - :literal:`"policy"`
      - observation
      - action
      - :ref:`Gaussian <models_gaussian>` / :ref:`MultivariateGaussian <models_multivariate_gaussian>`
    * - :math:`Q_{\phi 1}(s, a)`
      - Q1-network (critic 1)
      - :literal:`"critic_1"`
      - observation + action
      - 1
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_{\phi 2}(s, a)`
      - Q2-network (critic 2)
      - :literal:`"critic_2"`
      - observation + action
      - 1
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_{{\phi 1}_{target}}(s, a)`
      - Target Q1-network
      - :literal:`"target_critic_1"`
      - observation + action
      - 1
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_{{\phi 2}_{target}}(s, a)`
      - Target Q2-network
      - :literal:`"target_critic_2"`
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

.. autoclass:: skrl.agents.torch.sac.SAC_DEFAULT_CONFIG

.. autoclass:: skrl.agents.torch.sac.SAC
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__


.. autoclass:: skrl.agents.torch.sac.SAC_RNN
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.agents.jax.sac.SAC_DEFAULT_CONFIG

.. autoclass:: skrl.agents.jax.sac.SAC
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__
