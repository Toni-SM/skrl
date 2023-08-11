Adversarial Motion Priors (AMP)
===============================

AMP is a **model-free**, **stochastic** **on-policy** **policy gradient** algorithm (trained using a combination of GAIL and PPO) for adversarial learning of physics-based character animation. It enables characters to imitate diverse behaviors from large unstructured datasets, without the need for motion planners or other mechanisms for clip selection

Paper: `AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control <https://arxiv.org/abs/2104.02180>`_

.. raw:: html

    <br><hr>

Algorithm
---------

.. raw:: html

    <br>

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - policy (:math:`\pi_\theta`), value (:math:`V_\phi`) and discriminator (:math:`D_\psi`) function approximators
|   - states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next states (:math:`s'`), dones (:math:`d`)
|   - values (:math:`V`), next values (:math:`V'`), advantages (:math:`A`), returns (:math:`R`)
|   - log probabilities (:math:`logp`)
|   - loss (:math:`L`)
|   - reference motion dataset (:math:`M`), AMP replay buffer (:math:`B`)
|   - AMP states (:math:`s_{_{AMP}}`), reference motion states (:math:`s_{_{AMP}}^{^M}`), AMP states from replay buffer (:math:`s_{_{AMP}}^{^B}`)

.. raw:: html

    <br>

Learning algorithm
""""""""""""""""""

|
| :literal:`compute_gae(...)`
| :blue:`def` :math:`\;f_{GAE} (r, d, V, V') \;\rightarrow\; R, A:`
|     :math:`adv \leftarrow 0`
|     :math:`A \leftarrow \text{zeros}(r)`
|     :green:`# advantages computation`
|     **FOR** each reverse iteration :math:`i` up to the number of rows in :math:`r` **DO**
|         :math:`adv \leftarrow r_i - V_i \, +` :guilabel:`discount_factor` :math:`(V' \, +` :guilabel:`lambda` :math:`\neg d_i \; adv)`
|         :math:`A_i \leftarrow adv`
|     :green:`# returns computation`
|     :math:`R \leftarrow A + V`
|     :green:`# normalize advantages`
|     :math:`A \leftarrow \dfrac{A - \bar{A}}{A_\sigma + 10^{-8}}`

|
| :literal:`_update(...)`
| :green:`# update dataset of reference motions`
| collect reference motions of size :guilabel:`amp_batch_size` :math:`\rightarrow\;` :math:`\text{append}(M)`
| :green:`# compute combined rewards`
| :math:`r_D \leftarrow -log(\text{max}( 1 - \hat{y}(D_\psi(s_{_{AMP}})), \, 10^{-4})) \qquad` with :math:`\; \hat{y}(x) = \dfrac{1}{1 + e^{-x}}`
| :math:`r' \leftarrow` :guilabel:`task_reward_weight` :math:`r \, +` :guilabel:`style_reward_weight` :guilabel:`discriminator_reward_scale` :math:`r_D`
| :green:`# compute returns and advantages`
| :math:`R, A \leftarrow f_{GAE}(r', d, V, V')`
| :green:`# sample mini-batches from memory`
| [[:math:`s, a, logp, V, R, A, s_{_{AMP}}`]] :math:`\leftarrow` states, actions, log_prob, values, returns, advantages, AMP states
| [[:math:`s_{_{AMP}}^{^M}`]] :math:`\leftarrow` AMP states from :math:`M`
| **IF** :math:`B` is not empty **THEN**
|     [[:math:`s_{_{AMP}}^{^B}`]] :math:`\leftarrow` AMP states from :math:`B`
| **ELSE**
|     [[:math:`s_{_{AMP}}^{^B}`]] :math:`\leftarrow` [[:math:`s_{_{AMP}}`]]
| :green:`# learning epochs`
| **FOR** each learning epoch up to :guilabel:`learning_epochs` **DO**
|     :green:`# mini-batches loop`
|     **FOR** each mini-batch [:math:`s, a, logp, V, R, A, s_{_{AMP}}, s_{_{AMP}}^{^B}, s_{_{AMP}}^{^M}`] up to :guilabel:`mini_batches` **DO**
|         :math:`logp' \leftarrow \pi_\theta(s, a)`
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
|         :green:`# compute discriminator loss`
|         :math:`{logit}_{_{AMP}} \leftarrow D_\psi(s_{_{AMP}}) \qquad` with :math:`s_{_{AMP}}` of size :guilabel:`discriminator_batch_size`
|         :math:`{logit}_{_{AMP}}^{^B} \leftarrow D_\psi(s_{_{AMP}}^{^B}) \qquad` with :math:`s_{_{AMP}}^{^B}` of size :guilabel:`discriminator_batch_size`
|         :math:`{logit}_{_{AMP}}^{^M} \leftarrow D_\psi(s_{_{AMP}}^{^M}) \qquad` with :math:`s_{_{AMP}}^{^M}` of size :guilabel:`discriminator_batch_size`
|         :green:`# discriminator prediction loss`
|         :math:`L_{D_\psi} \leftarrow \dfrac{1}{2}(BCE({logit}_{_{AMP}}` ++ :math:`{logit}_{_{AMP}}^{^B}, \, 0) + BCE({logit}_{_{AMP}}^{^M}, \, 1))`
|              with :math:`\; BCE(x,y)=-\frac{1}{N} \sum_{i=1}^N [y \; log(\hat{y}) + (1-y) \, log(1-\hat{y})] \;` and :math:`\; \hat{y} = \dfrac{1}{1 + e^{-x}}`
|         :green:`# discriminator logit regularization`
|         :math:`L_{D_\psi} \leftarrow L_{D_\psi} +` :guilabel:`discriminator_logit_regularization_scale` :math:`\sum_{i=1}^N \text{flatten}(\psi_w[-1])^2`
|         :green:`# discriminator gradient penalty`
|         :math:`L_{D_\psi} \leftarrow L_{D_\psi} +` :guilabel:`discriminator_gradient_penalty_scale` :math:`\frac{1}{N} \sum_{i=1}^N \sum (\nabla_\psi {logit}_{_{AMP}}^{^M})^2`
|         :green:`# discriminator weight decay`
|         :math:`L_{D_\psi} \leftarrow L_{D_\psi} +` :guilabel:`discriminator_weight_decay_scale` :math:`\sum_{i=1}^N \text{flatten}(\psi_w)^2`
|         :green:`# optimization step`
|         reset :math:`\text{optimizer}_{\theta, \phi, \psi}`
|         :math:`\nabla_{\theta, \, \phi, \, \psi} (L^{clip}_{\pi_\theta} + {L}_{entropy} + L_{V_\phi} + L_{D_\psi})`
|         :math:`\text{clip}(\lVert \nabla_{\theta, \, \phi, \, \psi} \rVert)` with :guilabel:`grad_norm_clip`
|         step :math:`\text{optimizer}_{\theta, \phi, \psi}`
|     :green:`# update learning rate`
|     **IF** there is a :guilabel:`learning_rate_scheduler` **THEN**
|         step :math:`\text{scheduler}_{\theta, \phi, \psi} (\text{optimizer}_{\theta, \phi, \psi})`
| :green:`# update AMP repaly buffer`
| :math:`s_{_{AMP}} \rightarrow\;` :math:`\text{append}(B)`

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
                    :start-after: [torch-start-amp]
                    :end-before: [torch-end-amp]

.. raw:: html

    <br>

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../skrl/agents/torch/amp/amp.py
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
      - .. centered:: AMP observation
      - .. centered:: Observation
      - .. centered:: Action
    * - Discrete
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - Box
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - Dict
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`

.. raw:: html

    <br>

Models
^^^^^^

The implementation uses 1 stochastic (continuous) and 2 deterministic function approximators. These function approximators (models) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`models`

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
    * - :math:`D_\psi(s_{_{AMP}})`
      - Discriminator
      - :literal:`"discriminator"`
      - AMP observation
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
      - \-
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.agents.torch.amp.AMP_DEFAULT_CONFIG

.. autoclass:: skrl.agents.torch.amp.AMP
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__
