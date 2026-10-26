:tocdepth: 4

Heterogeneous-Agent Trust Region Policy Optimization (HATRPO)
=============================================================

HATRPO is a **model-free**, **stochastic** **on-policy** **policy gradient** CTDE (centralized training,
decentralized execution) **multi-agent** algorithm for cooperative tasks that updates the agents' policies
sequentially (one agent after the other, in a random order). Each policy update takes into account the updates of the
previously updated agents, without requiring the agents to share parameters (agents can be heterogeneous).
It follows the same sequential scheme as :doc:`HAPPO <happo>`, but each policy is updated with a
:doc:`TRPO <../agents/trpo>` step: a natural gradient step on the surrogate objective (weighted by the previously
updated agents' probability ratios) followed by a backtracking line search under a KL divergence constraint.

Heterogeneous-agent trust region theory shows that sequential trust-region updates of this kind improve the joint return monotonically when
the trust-region problem of each agent is solved exactly. The implemented step only approximates it (sample estimates,
a quadratic approximation of the KL divergence solved with a few conjugate gradient iterations, and a line search), so
the improvement is not guaranteed in practice.

**Paper:** `Heterogeneous-Agent Reinforcement Learning <https://jmlr.org/papers/v25/23-0488.html>`_.

|br| |hr|

Algorithm
---------

| For each iteration do:
|     :math:`\bullet \;` Collect, in a rollout memory per agent, a set of states :math:`s`, actions :math:`a`, rewards :math:`r`, terminated :math:`d_{_{end}}`, truncated :math:`d_{_{timeout}}`, log probabilities :math:`logp` and values :math:`V` on policy using :math:`\pi_{\theta_k}` and :math:`V_\phi`
|     :math:`\bullet \;` Estimate returns :math:`R` and advantages :math:`A` using Generalized Advantage Estimation (GAE(:math:`\lambda`)) from the collected data [:math:`r, d_{_{end}} \lor d_{_{timeout}}, V`]
|     :math:`\bullet \;` Initialize the factor :math:`M \leftarrow 1`
|     For each agent :math:`k`, in a random order, do:
|         :math:`\bullet \;` Compute the gradient :math:`g` of the surrogate objective, weighted by the factor, with :math:`ratio` as the probability ratio between the action under the current policy and the action under the previous policy: :math:`L_{\pi_{\theta_k}} = \mathbb{E}[M \, A \; ratio]`
|         :math:`\bullet \;` Compute the search direction :math:`\; x \approx H^{-1}g \;` using the conjugate gradient method, where :math:`H` is the Hessian of the :math:`KL` divergence with respect to the policy parameters :math:`\theta_k`
|         :math:`\bullet \;` Compute the maximal (full) step length :math:`\; \beta = \sqrt{\dfrac{2 \delta}{x^T H x}} x \;` where :math:`\delta` is the desired (maximum) :math:`KL` divergence
|         :math:`\bullet \;` Perform a backtracking line search with exponential decay to find the final policy update :math:`\; \theta_k' = \theta_k + \alpha \; \beta \;` ensuring improvement of the surrogate objective and satisfaction of the :math:`KL` divergence constraint
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
| :literal:`surrogate_loss(...)`
| :blue:`def` :math:`\;f_{Loss} (\pi_{\theta_k}, o, a, logp, A, M) \;\rightarrow\; L_{\pi_{\theta_k}}:`
|     :math:`logp' \leftarrow \pi_{\theta_k}(o, a)`
|     :math:`L_{\pi_{\theta_k}} \leftarrow \frac{1}{N} \sum_{i=1}^N M \, A \; e^{(logp' - logp)}`

|
| :literal:`conjugate_gradient(...)` (See `conjugate gradient method <https://en.wikipedia.org/wiki/Conjugate_gradient_method#As_an_iterative_method>`_)
| :blue:`def` :math:`\;f_{CG} (\pi_{\theta_k}, o, b) \;\rightarrow\; x:`
|     :math:`x \leftarrow \text{zeros}(b)`
|     :math:`r \leftarrow b`
|     :math:`p \leftarrow b`
|     :math:`rr_{old} \leftarrow r \cdot r`
|     **FOR** each iteration up to :guilabel:`conjugate_gradient_steps` **DO**
|         :math:`\alpha \leftarrow \dfrac{rr_{old}}{p \cdot f_{Ax}(\pi_{\theta_k}, o, p)}`
|         :math:`x \leftarrow x + \alpha \; p`
|         :math:`r \leftarrow r - \alpha \; f_{Ax}(\pi_{\theta_k}, o, p)`
|         :math:`rr_{new} \leftarrow r \cdot r`
|         **IF** :math:`rr_{new} <` residual tolerance **THEN**
|             **BREAK LOOP**
|         :math:`p \leftarrow r + \dfrac{rr_{new}}{rr_{old}} \; p`
|         :math:`rr_{old} \leftarrow rr_{new}`

|
| :literal:`fisher_vector_product(...)` (See `fisher vector product in TRPO <https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/>`_)
| :blue:`def` :math:`\;f_{Ax} (\pi_{\theta_k}, o, v) \;\rightarrow\; hv:`
|     :math:`kl \leftarrow f_{KL}(\pi_{\theta_k}, \pi_{\theta_k}, o)`
|     :math:`g_{kl} \leftarrow \nabla_{\theta_k} kl`
|     :math:`g_{kl_{flat}} \leftarrow \text{flatten}(g_{kl})`
|     :math:`g_{hv} \leftarrow \nabla_{\theta_k} (g_{kl_{flat}} \; v)`
|     :math:`g_{hv_{flat}} \leftarrow \text{flatten}(g_{hv})`
|     :math:`hv \leftarrow g_{hv_{flat}} +` :guilabel:`damping` :math:`v`

|
| :literal:`kl_divergence(...)` (See `Kullback–Leibler divergence for normal distribution <https://en.wikipedia.org/wiki/Normal_distribution#Other_properties>`_)
| :blue:`def` :math:`\;f_{KL} (\pi_{\theta 1}, \pi_{\theta 2}, o) \;\rightarrow\; kl:`
|     :math:`\mu_1, \log\sigma_1 \leftarrow \pi_{\theta 1}(o)`
|     :math:`\mu_2, \log\sigma_2 \leftarrow \pi_{\theta 2}(o)`
|     :math:`kl \leftarrow \log\sigma_2 - \log\sigma_1 + \frac{1}{2} \dfrac{(e^{\log\sigma_1})^2 + (\mu_1 - \mu_2)^2}{(e^{\log\sigma_2})^2} - \frac{1}{2}`
|     :math:`kl \leftarrow \frac{1}{N} \sum_{i=1}^N \, (\sum_{dim} kl)`

|
| :literal:`update(...)`
| :green:`# compute returns and advantages (for a shared value function, once from the team reward: the mean of the agents' rewards)`
| :math:`V_{_{last}}' \leftarrow V_\phi(s'_{_{shared}})`
| :math:`R, A \leftarrow f_{GAE}(r, d_{_{end}} \lor d_{_{timeout}}, V, V_{_{last}}')`
| :green:`# sequential policy updates`
| :math:`M \leftarrow 1`
| **FOR** each agent :math:`k` in a random permutation of the agents (or in order, if :guilabel:`fixed_order` is enabled) **DO**
|     :math:`logp_{_{old}} \leftarrow \pi_{\theta_k}(o_k, a_k)` for all the actions in the agent's memory
|     :green:`# sample all from the agent's memory`
|     [[:math:`o, a, logp, A, M`]] :math:`\leftarrow` observations, actions, log_prob, advantages, factor
|     :green:`# compute policy loss gradient`
|     :math:`L_{\pi_{\theta_k}} \leftarrow f_{Loss}(\pi_{\theta_k}, o, a, logp, A, M)`
|     :math:`g \leftarrow \nabla_{\theta_k} L_{\pi_{\theta_k}}`
|     :math:`g_{_{flat}} \leftarrow \text{flatten}(g)`
|     :green:`# compute the search direction using the conjugate gradient algorithm`
|     :math:`search_{direction} \leftarrow f_{CG}(\pi_{\theta_k}, o, g_{_{flat}})`
|     :green:`# compute step size and full step`
|     :math:`xHx \leftarrow search_{direction} \; f_{Ax}(\pi_{\theta_k}, o, search_{direction})`
|     :math:`step_{size} \leftarrow \sqrt{\dfrac{2 \, \delta}{xHx}} \qquad` with :math:`\; \delta` as :guilabel:`max_kl_divergence`
|     :math:`\beta \leftarrow step_{size} \; search_{direction}`
|     :green:`# backtracking line search`
|     :math:`flag_{restore} \leftarrow \text{True}`
|     :math:`\pi_{\theta_{k, backup}} \leftarrow \pi_{\theta_k}`
|     :math:`\theta \leftarrow \text{get_parameters}(\pi_{\theta_k})`
|     :math:`I_{expected} \leftarrow g_{_{flat}} \; \beta`
|     **FOR** :math:`\alpha \leftarrow` :guilabel:`step_fraction` :math:`\cdot` :guilabel:`backtrack_ratio` :math:`^i \;` with :math:`i = 0, 1, 2, ...` up to :guilabel:`max_backtrack_steps` **DO**
|         :math:`\theta_{new} \leftarrow \theta + \alpha \; \beta`
|         :math:`\pi_{\theta_k} \leftarrow \text{set_parameters}(\theta_{new})`
|         :math:`kl \leftarrow f_{KL}(\pi_{\theta_{k, backup}}, \pi_{\theta_k}, o)`
|         :math:`L \leftarrow f_{Loss}(\pi_{\theta_k}, o, a, logp, A, M)`
|         **IF** :math:`kl < \delta` **AND** :math:`L - L_{\pi_{\theta_k}} > 0` **AND** :math:`\dfrac{L - L_{\pi_{\theta_k}}}{\alpha \; I_{expected}} >` :guilabel:`accept_ratio` **THEN**
|             :math:`flag_{restore} \leftarrow \text{False}`
|             **BREAK LOOP**
|     **IF** :math:`flag_{restore}` **THEN**
|         :math:`\pi_{\theta_k} \leftarrow \pi_{\theta_{k, backup}}`
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
|         **IF** :guilabel:`value_loss_type` is ``"huber"`` **THEN**
|             :math:`L_{V_\phi} \leftarrow` :guilabel:`value_loss_scale` :math:`\frac{1}{N} \sum_{i=1}^N \text{Huber}_\delta(R - V_{_{predicted}}) \qquad` with :math:`\delta` as :guilabel:`huber_delta`
|         **ELSE**
|             :math:`L_{V_\phi} \leftarrow` :guilabel:`value_loss_scale` :math:`\frac{1}{N} \sum_{i=1}^N (R - V_{_{predicted}})^2`
|         :green:`# optimization step`
|         reset :math:`\text{optimizer}_\phi`
|         :math:`\nabla_\phi L_{V_\phi}`
|         :math:`\text{clip}(\lVert \nabla_\phi \rVert)` with :guilabel:`grad_norm_clip`
|         step :math:`\text{optimizer}_\phi`
|     :green:`# update learning rate`
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
                    :start-after: [start-hatrpo-torch]
                    :end-before: [end-hatrpo-torch]

|

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note::

    Each policy is updated with a single TRPO step on all the data in its agent's memory. Therefore,
    :guilabel:`learning_epochs`, :guilabel:`mini_batches`, :guilabel:`learning_rate`, :guilabel:`learning_rate_scheduler`
    and :guilabel:`grad_norm_clip` only apply to the value update. The value learning rate scheduler is stepped as in
    :doc:`TRPO <../agents/trpo>`, without any KL divergence (a
    :py:class:`~skrl.resources.schedulers.torch.kl_adaptive.KLAdaptiveLR` scheduler keeps the initial learning rate).

.. list-table::
    :header-rows: 1

    * - Dataclass
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - ``HATRPO_CFG``
      - :py:class:`~skrl.multi_agents.torch.hatrpo.HATRPO_CFG`
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
      - .. centered:: :math:`\square`
    * - MultiDiscrete
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - Box
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - Dict
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

|

Models
^^^^^^

The implementation uses 1 stochastic (continuous) and 1 deterministic function approximator per agent.
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
      - :ref:`Gaussian <models_gaussian>`
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

.. note::

    The policies are copied when the agent is instantiated (the copies are used by the line search).
    Models with lazy modules (e.g.: those created with the :doc:`model instantiators <../utils/model_instantiators>`)
    must therefore be initialized (e.g.: by calling their :literal:`init_state_dict` method) before instantiating
    the agent.

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
      - \-
      - .. centered:: :math:`\square`
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

.. automodule:: skrl.multi_agents.torch.hatrpo
.. autosummary::
    :nosignatures:

    HATRPO_CFG
    HATRPO

.. autoclass:: skrl.multi_agents.torch.hatrpo.HATRPO_CFG
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.multi_agents.torch.hatrpo.HATRPO
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
