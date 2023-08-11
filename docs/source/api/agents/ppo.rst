Proximal Policy Optimization (PPO)
==================================

PPO is a **model-free**, **stochastic** **on-policy** **policy gradient** algorithm that alternates between sampling data through interaction with the environment, and optimizing a *surrogate* objective function while avoiding that the new policy does not move too far away from the old one

Paper: `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_

.. raw:: html

    <br><hr>

Algorithm
---------

| For each iteration do:
|     :math:`\bullet \;` Collect, in a rollout memory, a set of states :math:`s`, actions :math:`a`, rewards :math:`r`, dones :math:`d`, log probabilities :math:`logp` and values :math:`V` on policy using :math:`\pi_\theta` and :math:`V_\phi`
|     :math:`\bullet \;` Estimate returns :math:`R` and advantages :math:`A` using Generalized Advantage Estimation (GAE(:math:`\lambda`)) from the collected data [:math:`r, d, V`]
|     :math:`\bullet \;` Compute the entropy loss :math:`{L}_{entropy}`
|     :math:`\bullet \;` Compute the clipped surrogate objective (policy loss) with :math:`ratio` as the probability ratio between the action under the current policy and the action under the previous policy: :math:`L^{clip}_{\pi_\theta} = \mathbb{E}[\min(A \; ratio, A \; \text{clip}(ratio, 1-c, 1+c))]`
|     :math:`\bullet \;` Compute the value loss :math:`L_{V_\phi}` as the mean squared error (MSE) between the predicted values :math:`V_{_{predicted}}` and the estimated returns :math:`R`
|     :math:`\bullet \;` Optimize the total loss :math:`L = L^{clip}_{\pi_\theta} - c_1 \, L_{V_\phi} + c_2 \, {L}_{entropy}`

.. raw:: html

    <br>

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - policy function approximator (:math:`\pi_\theta`), value function approximator (:math:`V_\phi`)
|   - states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next states (:math:`s'`), dones (:math:`d`)
|   - values (:math:`V`), advantages (:math:`A`), returns (:math:`R`)
|   - log probabilities (:math:`logp`)
|   - loss (:math:`L`)

.. raw:: html

    <br>

Learning algorithm
""""""""""""""""""

|
| :literal:`compute_gae(...)`
| :blue:`def` :math:`\;f_{GAE} (r, d, V, V_{_{last}}') \;\rightarrow\; R, A:`
|     :math:`adv \leftarrow 0`
|     :math:`A \leftarrow \text{zeros}(r)`
|     :green:`# advantages computation`
|     **FOR** each reverse iteration :math:`i` up to the number of rows in :math:`r` **DO**
|         **IF** :math:`i` is not the last row of :math:`r` **THEN**
|             :math:`V_i' = V_{i+1}`
|         **ELSE**
|             :math:`V_i' \leftarrow V_{_{last}}'`
|         :math:`adv \leftarrow r_i - V_i \, +` :guilabel:`discount_factor` :math:`\neg d_i \; (V_i' \, -` :guilabel:`lambda` :math:`adv)`
|         :math:`A_i \leftarrow adv`
|     :green:`# returns computation`
|     :math:`R \leftarrow A + V`
|     :green:`# normalize advantages`
|     :math:`A \leftarrow \dfrac{A - \bar{A}}{A_\sigma + 10^{-8}}`

|
| :literal:`_update(...)`
| :green:`# compute returns and advantages`
| :math:`V_{_{last}}' \leftarrow V_\phi(s')`
| :math:`R, A \leftarrow f_{GAE}(r, d, V, V_{_{last}}')`
| :green:`# sample mini-batches from memory`
| [[:math:`s, a, logp, V, R, A`]] :math:`\leftarrow` states, actions, log_prob, values, returns, advantages
| :green:`# learning epochs`
| **FOR** each learning epoch up to :guilabel:`learning_epochs` **DO**
|     :green:`# mini-batches loop`
|     **FOR** each mini-batch [:math:`s, a, logp, V, R, A`] up to :guilabel:`mini_batches` **DO**
|         :math:`logp' \leftarrow \pi_\theta(s, a)`
|         :green:`# compute approximate KL divergence`
|         :math:`ratio \leftarrow logp' - logp`
|         :math:`KL_{_{divergence}} \leftarrow \frac{1}{N} \sum_{i=1}^N ((e^{ratio} - 1) - ratio)`
|         :green:`# early stopping with KL divergence`
|         **IF** :math:`KL_{_{divergence}} >` :guilabel:`kl_threshold` **THEN**
|             **BREAK LOOP**
|         :green:`# compute entropy loss`
|         **IF** entropy computation is enabled **THEN**
|             :math:`{L}_{entropy} \leftarrow \, -` :guilabel:`entropy_loss_scale` :math:`\frac{1}{N} \sum_{i=1}^N \pi_{\theta_{entropy}}`
|         **ELSE**
|             :math:`{L}_{entropy} \leftarrow 0`
|         :green:`# compute policy loss`
|         :math:`ratio \leftarrow e^{logp' - logp}`
|         :math:`L_{_{surrogate}} \leftarrow A \; ratio`
|         :math:`L_{_{clipped\,surrogate}} \leftarrow A \; \text{clip}(ratio, 1 - c, 1 + c) \qquad` with :math:`c` as :guilabel:`ratio_clip`
|         :math:`L^{clip}_{\pi_\theta} \leftarrow - \frac{1}{N} \sum_{i=1}^N \min(L_{_{surrogate}}, L_{_{clipped\,surrogate}})`
|         :green:`# compute value loss`
|         :math:`V_{_{predicted}} \leftarrow V_\phi(s)`
|         **IF** :guilabel:`clip_predicted_values` is enabled **THEN**
|             :math:`V_{_{predicted}} \leftarrow V + \text{clip}(V_{_{predicted}} - V, -c, c) \qquad` with :math:`c` as :guilabel:`value_clip`
|         :math:`L_{V_\phi} \leftarrow` :guilabel:`value_loss_scale` :math:`\frac{1}{N} \sum_{i=1}^N (R - V_{_{predicted}})^2`
|         :green:`# optimization step`
|         reset :math:`\text{optimizer}_{\theta, \phi}`
|         :math:`\nabla_{\theta, \, \phi} (L^{clip}_{\pi_\theta} + {L}_{entropy} + L_{V_\phi})`
|         :math:`\text{clip}(\lVert \nabla_{\theta, \, \phi} \rVert)` with :guilabel:`grad_norm_clip`
|         step :math:`\text{optimizer}_{\theta, \phi}`
|     :green:`# update learning rate`
|     **IF** there is a :guilabel:`learning_rate_scheduler` **THEN**
|         step :math:`\text{scheduler}_{\theta, \phi} (\text{optimizer}_{\theta, \phi})`

.. raw:: html

    <br>

Usage
-----

.. note::

    Support for recurrent neural networks (RNN, LSTM, GRU and any other variant) is implemented in a separate file (:literal:`ppo_rnn.py`) to maintain the readability of the standard implementation (:literal:`ppo.py`)

.. tabs::

    .. tab:: Standard implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [torch-start-ppo]
                    :end-before: [torch-end-ppo]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [jax-start-ppo]
                    :end-before: [jax-end-ppo]

    .. tab:: RNN implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. note::

                    When using recursive models it is necessary to override their :literal:`.get_specification()` method. Visit each model's documentation for more details

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [torch-start-ppo-rnn]
                    :end-before: [torch-end-ppo-rnn]

.. raw:: html

    <br>

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../skrl/agents/torch/ppo/ppo.py
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
      - .. centered:: :math:`\blacksquare`
    * - Dict
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

.. raw:: html

    <br>

Models
^^^^^^

The implementation uses 1 stochastic (discrete or continuous) and 1 deterministic function approximator. These function approximators (models) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`models`

.. list-table::
    :header-rows: 1

    * - Notation
      - Concept
      - Key
      - Input shape
      - Output shape
      - Type
    * - :math:`\pi_\theta(s)`
      - Policy
      - :literal:`"policy"`
      - observation
      - action
      - :ref:`Categorical <models_categorical>` / :ref:`Gaussian <models_gaussian>` / :ref:`MultivariateGaussian <models_multivariate_gaussian>`
    * - :math:`V_\phi(s)`
      - Value
      - :literal:`"value"`
      - observation
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
      - for Policy and Value
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - RNN support
      - RNN, LSTM, GRU and any other variant
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.agents.torch.ppo.PPO_DEFAULT_CONFIG

.. autoclass:: skrl.agents.torch.ppo.PPO
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.agents.torch.ppo.PPO_RNN
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.agents.jax.ppo.PPO_DEFAULT_CONFIG

.. autoclass:: skrl.agents.jax.ppo.PPO
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__
