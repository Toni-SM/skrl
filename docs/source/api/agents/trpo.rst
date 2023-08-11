Trust Region Policy Optimization (TRPO)
=======================================

TRPO is a **model-free**, **stochastic** **on-policy** **policy gradient** algorithm that deploys an iterative procedure to optimize the policy, with guaranteed monotonic improvement

Paper: `Trust Region Policy Optimization <https://arxiv.org/abs/1502.05477>`_

.. raw:: html

    <br><hr>

Algorithm
---------

| For each iteration do
|     :math:`\bullet \;` Collect, in a rollout memory, a set of states :math:`s`, actions :math:`a`, rewards :math:`r`, dones :math:`d`, log probabilities :math:`logp` and values :math:`V` on policy using :math:`\pi_\theta` and :math:`V_\phi`
|     :math:`\bullet \;` Estimate returns :math:`R` and advantages :math:`A` using Generalized Advantage Estimation (GAE(:math:`\lambda`)) from the collected data [:math:`r, d, V`]
|     :math:`\bullet \;` Compute the surrogate objective (policy loss) gradient :math:`g` and the Hessian :math:`H` of :math:`KL` divergence with respect to the policy parameters :math:`\theta`
|     :math:`\bullet \;` Compute the search direction :math:`\; x \approx H^{-1}g \;` using the conjugate gradient method
|     :math:`\bullet \;` Compute the maximal (full) step length :math:`\; \beta = \sqrt{\dfrac{2 \delta}{x^T H x}} x \;` where :math:`\delta` is the desired (maximum) :math:`KL` divergence and :math:`\; \sqrt{\frac{2 \delta}{x^T H x}} \;` is the step size
|     :math:`\bullet \;` Perform a backtracking line search with exponential decay to find the final policy update :math:`\; \theta_{new} = \theta + \alpha \; \beta \;` ensuring improvement of the surrogate objective and satisfaction of the :math:`KL` divergence constraint
|     :math:`\bullet \;` Update the value function :math:`V_\phi` using the computed returns :math:`R`

.. raw:: html

    <br>

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

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
| :literal:`surrogate_loss(...)`
| :blue:`def` :math:`\;f_{Loss} (\pi_\theta, s, a, logp, A) \;\rightarrow\; L_{\pi_\theta}:`
|     :math:`logp' \leftarrow \pi_\theta(s, a)`
|     :math:`L_{\pi_\theta} \leftarrow \frac{1}{N} \sum_{i=1}^N A \; e^{(logp' - logp)}`

|
| :literal:`conjugate_gradient(...)` (See `conjugate gradient method <https://en.wikipedia.org/wiki/Conjugate_gradient_method#As_an_iterative_method>`_)
| :blue:`def` :math:`\;f_{CG} (\pi_\theta, s, b) \;\rightarrow\; x:`
|     :math:`x \leftarrow \text{zeros}(b)`
|     :math:`r \leftarrow b`
|     :math:`p \leftarrow b`
|     :math:`rr_{old} \leftarrow r \cdot r`
|     **FOR** each iteration up to :guilabel:`conjugate_gradient_steps` **DO**
|         :math:`\alpha \leftarrow \dfrac{rr_{old}}{p \cdot f_{Ax}(\pi_\theta, s, b)}`
|         :math:`x \leftarrow x + \alpha \; p`
|         :math:`r \leftarrow r - \alpha \; f_{Ax}(\pi_\theta, s)`
|         :math:`rr_{new} \leftarrow r \cdot r`
|         **IF** :math:`rr_{new} <` residual tolerance **THEN**
|             **BREAK LOOP**
|         :math:`p \leftarrow r + \dfrac{rr_{new}}{rr_{old}} \; p`
|         :math:`rr_{old} \leftarrow rr_{new}`

|
| :literal:`fisher_vector_product(...)` (See `fisher vector product in TRPO <https://www.telesens.co/2018/06/09/efficiently-computing-the-fisher-vector-product-in-trpo/>`_)
| :blue:`def` :math:`\;f_{Ax} (\pi_\theta, s, v) \;\rightarrow\; hv:`
|     :math:`kl \leftarrow f_{KL}(\pi_\theta, \pi_\theta, s)`
|     :math:`g_{kl} \leftarrow \nabla_\theta kl`
|     :math:`g_{kl_{flat}} \leftarrow \text{flatten}(g_{kl})`
|     :math:`g_{hv} \leftarrow \nabla_\theta (g_{kl_{flat}} \; v)`
|     :math:`g_{hv_{flat}} \leftarrow \text{flatten}(g_{hv})`
|     :math:`hv \leftarrow g_{hv_{flat}} +` :guilabel:`damping` :math:`v`

|
| :literal:`kl_divergence(...)` (See `Kullback–Leibler divergence for normal distribution <https://en.wikipedia.org/wiki/Normal_distribution#Other_properties>`_)
| :blue:`def` :math:`\;f_{KL} (\pi_{\theta 1}, \pi_{\theta 2}, s) \;\rightarrow\; kl:`
|     :math:`\mu_1, \log\sigma_1 \leftarrow \pi_{\theta 1}(s)`
|     :math:`\mu_2, \log\sigma_2 \leftarrow \pi_{\theta 2}(s)`
|     :math:`kl \leftarrow \log\sigma_1 - \log\sigma_2 + \frac{1}{2} \dfrac{(e^{\log\sigma_1})^2 + (\mu_1 - \mu_2)^2}{(e^{\log\sigma_2})^2} - \frac{1}{2}`
|     :math:`kl \leftarrow \frac{1}{N} \sum_{i=1}^N \, (\sum_{dim} kl)`

|
| :literal:`_update(...)`
| :green:`# compute returns and advantages`
| :math:`V_{_{last}}' \leftarrow V_\phi(s')`
| :math:`R, A \leftarrow f_{GAE}(r, d, V, V_{_{last}}')`
| :green:`# sample all from memory`
| [[:math:`s, a, logp, A`]] :math:`\leftarrow` states, actions, log_prob, advantages
| :green:`# compute policy loss gradient`
| :math:`L_{\pi_\theta} \leftarrow f_{Loss}(\pi_\theta, s, a, logp, A)`
| :math:`g \leftarrow \nabla_{\theta} L_{\pi_\theta}`
| :math:`g_{_{flat}} \leftarrow \text{flatten}(g)`
| :green:`# compute the search direction using the conjugate gradient algorithm`
| :math:`search_{direction} \leftarrow f_{CG}(\pi_\theta, s, g_{_{flat}})`
| :green:`# compute step size and full step`
| :math:`xHx \leftarrow search_{direction} \; f_{Ax}(\pi_\theta, s, search_{direction})`
| :math:`step_{size} \leftarrow \sqrt{\dfrac{2 \, \delta}{xHx}} \qquad` with :math:`\; \delta` as :guilabel:`max_kl_divergence`
| :math:`\beta \leftarrow step_{size} \; search_{direction}`
| :green:`# backtracking line search`
| :math:`flag_{restore} \leftarrow \text{True}`
| :math:`\pi_{\theta_{backup}} \leftarrow \pi_\theta`
| :math:`\theta \leftarrow \text{get_parameters}(\pi_\theta)`
| :math:`I_{expected} \leftarrow g_{_{flat}} \; \beta`
| **FOR** :math:`\alpha \leftarrow (0.5` :guilabel:`step_fraction` :math:`)^i \;` with :math:`i = 0, 1, 2, ...` up to :guilabel:`max_backtrack_steps` **DO**
|     :math:`\theta_{new} \leftarrow \theta + \alpha \; \beta`
|     :math:`\pi_\theta \leftarrow \text{set_parameters}(\theta_{new})`
|     :math:`I_{expected} \leftarrow \alpha \; I_{expected}`
|     :math:`kl \leftarrow f_{KL}(\pi_{\theta_{backup}}, \pi_\theta, s)`
|     :math:`L \leftarrow f_{Loss}(\pi_\theta, s, a, logp, A)`
|     **IF** :math:`kl < \delta` **AND** :math:`\dfrac{L - L_{\pi_\theta}}{I_{expected}} >` :guilabel:`accept_ratio` **THEN**
|         :math:`flag_{restore} \leftarrow \text{False}`
|         **BREAK LOOP**
| **IF** :math:`flag_{restore}` **THEN**
|     :math:`\pi_\theta \leftarrow \pi_{\theta_{backup}}`
| :green:`# sample mini-batches from memory`
| [[:math:`s, R`]] :math:`\leftarrow` states, returns
| :green:`# learning epochs`
| **FOR** each learning epoch up to :guilabel:`learning_epochs` **DO**
|     :green:`# mini-batches loop`
|     **FOR** each mini-batch [:math:`s, R`] up to :guilabel:`mini_batches` **DO**
|          :green:`# compute value loss`
|          :math:`V' \leftarrow V_\phi(s)`
|          :math:`L_{V_\phi} \leftarrow` :guilabel:`value_loss_scale` :math:`\frac{1}{N} \sum_{i=1}^N (R - V')^2`
|          :green:`# optimization step (value)`
|          reset :math:`\text{optimizer}_\phi`
|          :math:`\nabla_{\phi} L_{V_\phi}`
|          :math:`\text{clip}(\lVert \nabla_{\phi} \rVert)` with :guilabel:`grad_norm_clip`
|          step :math:`\text{optimizer}_\phi`
|     :green:`# update learning rate`
|     **IF** there is a :guilabel:`learning_rate_scheduler` **THEN**
|         step :math:`\text{scheduler}_\phi(\text{optimizer}_\phi)`

.. raw:: html

    <br>

Usage
-----

.. note::

    Support for recurrent neural networks (RNN, LSTM, GRU and any other variant) is implemented in a separate file (:literal:`trpo_rnn.py`) to maintain the readability of the standard implementation (:literal:`trpo.py`)

.. tabs::

    .. tab:: Standard implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [torch-start-trpo]
                    :end-before: [torch-end-trpo]

    .. tab:: RNN implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. note::

                    When using recursive models it is necessary to override their :literal:`.get_specification()` method. Visit each model's documentation for more details

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [torch-start-trpo-rnn]
                    :end-before: [torch-end-trpo-rnn]

.. raw:: html

    <br>

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../skrl/agents/torch/trpo/trpo.py
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

The implementation uses 1 stochastic and 1 deterministic function approximator. These function approximators (models) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`models`

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
      - :ref:`Gaussian <models_gaussian>` / :ref:`MultivariateGaussian <models_multivariate_gaussian>`
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

.. autoclass:: skrl.agents.torch.trpo.TRPO_DEFAULT_CONFIG

.. autoclass:: skrl.agents.torch.trpo.TRPO
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.agents.torch.trpo.TRPO_RNN
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__
