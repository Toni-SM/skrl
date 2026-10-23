:tocdepth: 4

Heterogeneous-Agent Advantage Actor-Critic (HAA2C)
===================================================

HAA2C is a **model-free**, **stochastic** **on-policy** **policy gradient** CTDE (centralized training,
decentralized execution) **multi-agent** algorithm for cooperative tasks that updates the agents' policies
sequentially (one agent after the other, in a random order). Each policy update takes into account the updates of the
previously updated agents, without requiring the agents to share parameters (agents can be heterogeneous).
It follows the same sequential scheme as :doc:`HAPPO <happo>`, but its policy loss is the advantage-weighted
probability ratio without clipping, and the policies are optimized without KL early stopping.

**Paper:** `Heterogeneous-Agent Reinforcement Learning <https://jmlr.org/papers/v25/23-0488.html>`_.

|br| |hr|

Algorithm
---------

| For each iteration do:
|     :math:`\bullet \;` Collect, in a rollout memory per agent, a set of states :math:`s`, actions :math:`a`, rewards :math:`r`, terminated :math:`d_{_{end}}`, truncated :math:`d_{_{timeout}}`, log probabilities :math:`logp` and values :math:`V` on policy using :math:`\pi_{\theta_k}` and :math:`V_\phi`
|     :math:`\bullet \;` Estimate returns :math:`R` and advantages :math:`A` using Generalized Advantage Estimation (GAE(:math:`\lambda`)) from the collected data [:math:`r, d_{_{end}} \lor d_{_{timeout}}, V`]
|     :math:`\bullet \;` Initialize the factor :math:`M \leftarrow 1`
|     For each agent :math:`k`, in a random order, do:
|         :math:`\bullet \;` Compute the entropy loss :math:`{L}_{entropy}`
|         :math:`\bullet \;` Compute the policy loss, weighted by the factor, with :math:`ratio` as the probability ratio between the action under the current policy and the action under the previous policy: :math:`L_{\pi_{\theta_k}} = -\mathbb{E}[M \, A \; ratio]`
|         :math:`\bullet \;` Optimize the policy loss :math:`L = L_{\pi_{\theta_k}} + c_2 \, {L}_{entropy}`
|         :math:`\bullet \;` Update the factor with the probability ratio between the updated and the previous policy of the agent: :math:`M \leftarrow M \, \dfrac{\pi_{\theta_k'}(a_k|o_k)}{\pi_{\theta_k}(a_k|o_k)}`
|     :math:`\bullet \;` Compute the value loss :math:`L_{V_\phi}` as the mean squared error (MSE) between the predicted values :math:`V_{_{predicted}}` and the estimated returns :math:`R`
|     :math:`\bullet \;` Optimize the value loss :math:`L = c_1 \, L_{V_\phi}`

|

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - policy function approximator of the agent :math:`k` (:math:`\pi_{\theta_k}`), value function approximator (:math:`V_\phi`)
|   - observations (:math:`o`), states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next states (:math:`s'`), terminated (:math:`d_{_{end}}`), truncated (:math:`d_{_{timeout}}`)
|   - shared states (:math:`s_{_{shared}}`), shared next states (:math:`s'_{_{shared}}`)
|   - values (:math:`V`), advantages (:math:`A`), returns (:math:`R`)
|   - log probabilities (:math:`logp`), factor (:math:`M`)
|   - loss (:math:`L`)

|

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
|         :math:`adv \leftarrow r_i - V_i \, +` :guilabel:`discount_factor` :math:`\neg d_i \; (V_i' \, +` :guilabel:`lambda` :math:`adv)`
|         :math:`A_i \leftarrow adv`
|     :green:`# returns computation`
|     :math:`R \leftarrow A + V`
|     :green:`# normalize advantages`
|     :math:`A \leftarrow \dfrac{A - \bar{A}}{A_\sigma + 10^{-8}}`

|
| :literal:`update(...)`
| :green:`# compute returns and advantages (for a shared value function, once from the team reward: the mean of the agents' rewards)`
| :math:`V_{_{last}}' \leftarrow V_\phi(s'_{_{shared}})`
| :math:`R, A \leftarrow f_{GAE}(r, d_{_{end}} \lor d_{_{timeout}}, V, V_{_{last}}')`
| :green:`# sequential policy updates`
| :math:`M \leftarrow 1`
| **FOR** each agent :math:`k` in a random permutation of the agents (or in order, if :guilabel:`fixed_order` is enabled) **DO**
|     :math:`logp_{_{old}} \leftarrow \pi_{\theta_k}(o_k, a_k)` for all the actions in the agent's memory
|     :green:`# sample mini-batches from the agent's memory`
|     [[:math:`o, a, logp, A, M`]] :math:`\leftarrow` observations, actions, log_prob, advantages, factor
|     :green:`# learning epochs`
|     **FOR** each learning epoch up to :guilabel:`learning_epochs` **DO**
|         :green:`# mini-batches loop`
|         **FOR** each mini-batch [:math:`o, a, logp, A, M`] up to :guilabel:`mini_batches` **DO**
|             :math:`logp' \leftarrow \pi_{\theta_k}(o, a)`
|             :green:`# compute approximate KL divergence (for a KL adaptive learning rate scheduler)`
|             :math:`ratio \leftarrow logp' - logp`
|             :math:`KL_{_{divergence}} \leftarrow \frac{1}{N} \sum_{i=1}^N ((e^{ratio} - 1) - ratio)`
|             :green:`# compute entropy loss`
|             **IF** entropy computation is enabled **THEN**
|                 :math:`{L}_{entropy} \leftarrow \, -` :guilabel:`entropy_loss_scale` :math:`\frac{1}{N} \sum_{i=1}^N \pi_{\theta_{k, entropy}}`
|             **ELSE**
|                 :math:`{L}_{entropy} \leftarrow 0`
|             :green:`# compute policy loss`
|             :math:`ratio \leftarrow e^{logp' - logp}`
|             :math:`L_{\pi_{\theta_k}} \leftarrow - \frac{1}{N} \sum_{i=1}^N M \, A \; ratio`
|             :green:`# optimization step`
|             reset :math:`\text{optimizer}_{\theta_k}`
|             :math:`\nabla_{\theta_k} (L_{\pi_{\theta_k}} + {L}_{entropy})`
|             :math:`\text{clip}(\lVert \nabla_{\theta_k} \rVert)` with :guilabel:`grad_norm_clip`
|             step :math:`\text{optimizer}_{\theta_k}`
|         :green:`# update learning rate (a KL adaptive scheduler uses the KL divergence of the epoch)`
|         **IF** there is a :guilabel:`learning_rate_scheduler` **THEN**
|             step :math:`\text{scheduler}_{\theta_k} (\text{optimizer}_{\theta_k})`
|     :green:`# update the factor with the probability ratio between the updated and the previous policy`
|     :math:`M \leftarrow M \, e^{\pi_{\theta_k}(o_k, a_k) - logp_{_{old}}}` for all the actions in the agent's memory
| :green:`# value update (once for a shared value function)`
| :green:`# sample mini-batches from memory`
| [[:math:`s, V, R`]] :math:`\leftarrow` states, values, returns
| :green:`# learning epochs`
| **FOR** each learning epoch up to :guilabel:`learning_epochs` **DO**
|     :green:`# mini-batches loop`
|     **FOR** each mini-batch [:math:`s, V, R`] up to :guilabel:`mini_batches` **DO**
|         :green:`# compute value loss`
|         :math:`V_{_{predicted}} \leftarrow V_\phi(s_{_{shared}})`
|         **IF** :guilabel:`value_clip` :math:`> 0` **THEN**
|             :math:`V_{_{predicted}} \leftarrow V + \text{clip}(V_{_{predicted}} - V, -c, c) \qquad` with :math:`c` as :guilabel:`value_clip`
|         :math:`L_{V_\phi} \leftarrow` :guilabel:`value_loss_scale` :math:`\frac{1}{N} \sum_{i=1}^N (R - V_{_{predicted}})^2`
|         :green:`# optimization step`
|         reset :math:`\text{optimizer}_\phi`
|         :math:`\nabla_\phi L_{V_\phi}`
|         :math:`\text{clip}(\lVert \nabla_\phi \rVert)` with :guilabel:`grad_norm_clip`
|         step :math:`\text{optimizer}_\phi`
|     :green:`# update learning rate (a KL adaptive scheduler uses the agents' mean KL divergence)`
|     **IF** there is a :guilabel:`learning_rate_scheduler` **THEN**
|         step :math:`\text{scheduler}_\phi (\text{optimizer}_\phi)`

|

Usage
-----

.. tabs::

    .. tab:: Standard implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/multi_agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [start-haa2c-torch]
                    :end-before: [end-haa2c-torch]

|

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    :guilabel:`learning_epochs` and :guilabel:`mini_batches` apply to both the policy updates and the value update.

.. warning::

    If :guilabel:`learning_rate_scheduler` is a single :py:class:`~skrl.resources.schedulers.torch.kl_adaptive.KLAdaptiveLR`
    class (as in the :doc:`HAPPO <happo>` configuration), it is also applied to the value network, whose scheduler is
    stepped with the mean KL divergence of the agents' policies. Since the HAA2C policy loss is neither clipped nor
    stopped early, this divergence is usually large and drives the value learning rate to its minimum, so the value
    network practically stops learning. To adapt only the policy learning rate, provide a tuple of schedulers
    (policy, value), e.g.: :literal:`(KLAdaptiveLR, None)` with
    :literal:`learning_rate_scheduler_kwargs=({"kl_threshold": 0.008}, {})`.

.. list-table::
    :header-rows: 1

    * - Dataclass
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - ``HAA2C_CFG``
      - :py:class:`~skrl.multi_agents.torch.haa2c.HAA2C_CFG`
      -
      -

|

Spaces
^^^^^^

The implementation supports the following `Gymnasium spaces <https://gymnasium.farama.org/api/spaces>`_:

.. list-table::
    :header-rows: 1

    * - Gymnasium spaces
      - .. centered:: Observation
      - .. centered:: Action
    * - Discrete
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\blacksquare`
    * - MultiDiscrete
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\blacksquare`
    * - Box
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - Dict
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

|

Models
^^^^^^

The implementation uses 1 stochastic (discrete or continuous) and 1 deterministic function approximator per agent.
These function approximators (models) must be collected in a dictionary and passed to the constructor of the class
under the argument :literal:`models`.

.. list-table::
    :header-rows: 1

    * - Notation
      - Concept
      - Key
      - Input shape
      - Output shape
      - Type
    * - :math:`\pi_{\theta_k}(o)`
      - Policy
      - :literal:`"policy"`
      - observation
      - action
      - :ref:`Categorical <models_categorical>` /
        |br| :ref:`Multi-Categorical <models_multicategorical>` /
        |br| :ref:`Gaussian <models_gaussian>` /
        |br| :ref:`MultivariateGaussian <models_multivariate_gaussian>`
    * - :math:`V_\phi(s)`
      - Value
      - :literal:`"value"`
      - state
      - 1
      - :ref:`Deterministic <models_deterministic>`

.. note::

    The value model can be shared by all the agents or be specific to each agent:

    * **Shared value**: the same model instance is passed for every agent. It is trained once per update on the team
      reward (the mean of the agents' rewards), and the agents' episodes end when all the agents have terminated
      (or truncated). The agents' state spaces, as well as the value-related configuration (e.g.: discount factor,
      GAE lambda, value learning rate, state and value preprocessors), must be the same for all the agents.
    * **Per-agent value**: a distinct model instance is passed for every agent. Each value model is trained on its
      agent's reward.

    Since the value is updated after all the policies, the policy and value of an agent must be different models.
    Each agent must have its own policy instance (a policy shared by several agents raises a ``ValueError``).

.. warning::

    The policies' log probabilities must be sum-reduced per sample (shape ``(N, 1)``), e.g.: the
    :ref:`Gaussian <models_gaussian>` model's ``reduction="sum"`` (default). Otherwise, the probability ratios
    that make up the factor :math:`M` (the product of the previously updated agents' ratios) are wrong.

|

Features
^^^^^^^^

Support for advanced features is described in the following table:

.. list-table::
    :header-rows: 1

    * - Feature
      - Support and remarks
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - Shared model
      - Value shared by all the agents
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - RNN support
      - \-
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - Mixed precision
      - Automatic mixed precision
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - Distributed
      - Single Program Multi Data (SPMD) multi-GPU
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`

|

API
---

|

PyTorch
^^^^^^^

.. automodule:: skrl.multi_agents.torch.haa2c
.. autosummary::
    :nosignatures:

    HAA2C_CFG
    HAA2C

.. autoclass:: skrl.multi_agents.torch.haa2c.HAA2C_CFG
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.multi_agents.torch.haa2c.HAA2C
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
