Twin-Delayed DDPG (TD3)
=======================

TD3 is a **model-free**, **deterministic** **off-policy** **actor-critic** algorithm (based on DDPG) that relies on double Q-learning, target policy smoothing and delayed policy updates to address the problems introduced by overestimation bias in actor-critic algorithms 

Paper: `Addressing Function Approximation Error in Actor-Critic Methods <https://arxiv.org/abs/1802.09477>`_

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - policy function approximator (:math:`\mu_\theta`), critic function approximator (:math:`Q_\phi`)
|   - states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next states (:math:`s'`), dones (:math:`d`)
|   - loss (:math:`L`)

**Decision making** (:literal:`act(...)`)

| :math:`a \leftarrow \mu_\theta(s)`
| :math:`noise \leftarrow` sample :guilabel:`noise`
| :math:`scale \leftarrow (1 - \text{timestep} \;/` :guilabel:`timesteps` :math:`) \; (` :guilabel:`initial_scale` :math:`-` :guilabel:`final_scale` :math:`) \;+` :guilabel:`final_scale`
| :math:`a \leftarrow \text{clip}(a + noise * scale, {a}_{Low}, {a}_{High})`

**Learning algorithm** (:literal:`_update(...)`)

| :green:`# sample a batch from memory`
| [:math:`s, a, r, s', d`] :math:`\leftarrow` states, actions, rewards, next_states, dones of size :guilabel:`batch_size`
| :green:`# gradient steps`
| **FOR** each gradient step up to :guilabel:`gradient_steps` **DO**
|     :green:`# target policy smoothing`
|     :math:`a' \leftarrow \mu_{\theta_{target}}(s')`
|     :math:`noise \leftarrow \text{clip}(` :guilabel:`smooth_regularization_noise` :math:`, -c, c) \qquad` with :math:`c` as :guilabel:`smooth_regularization_clip` 
|     :math:`a' \leftarrow a' + noise`
|     :math:`a' \leftarrow \text{clip}(a', {a'}_{Low}, {a'}_{High})`
|     :green:`# compute target values`
|     :math:`Q_{1_{target}} \leftarrow Q_{{\phi 1}_{target}}(s', a')`
|     :math:`Q_{2_{target}} \leftarrow Q_{{\phi 2}_{target}}(s', a')`
|     :math:`Q_{_{target}} \leftarrow \text{min}(Q_{1_{target}}, Q_{2_{target}})`
|     :math:`y \leftarrow r \;+` :guilabel:`discount_factor` :math:`\neg d \; Q_{_{target}}`
|     :green:`# compute critic loss`
|     :math:`Q_1 \leftarrow Q_{\phi 1}(s, a)`
|     :math:`Q_2 \leftarrow Q_{\phi 2}(s, a)`
|     :math:`L_{Q_\phi} \leftarrow \frac{1}{N} \sum_{i=1}^N (Q_1 - y)^2 + \frac{1}{N} \sum_{i=1}^N (Q_2 - y)^2`
|     :green:`# optimization step (critic)`
|     reset :math:`\text{optimizer}_\phi`
|     :math:`\nabla_{\phi} L_{Q_\phi}`
|     step :math:`\text{optimizer}_\phi`
|     :green:`# delayed update`
|     **IF** it's time for the :guilabel:`policy_delay` update **THEN**
|         :green:`# compute policy (actor) loss`
|         :math:`a \leftarrow \mu_\theta(s)`
|         :math:`Q_1 \leftarrow Q_{\phi 1}(s, a)`
|         :math:`L_{\mu_\theta} \leftarrow - \frac{1}{N} \sum_{i=1}^N Q_1`
|         :green:`# optimization step (policy)`
|         reset :math:`\text{optimizer}_\theta`
|         :math:`\nabla_{\theta} L_{\mu_\theta}`
|         step :math:`\text{optimizer}_\theta`
|         :green:`# update target networks`
|         :math:`\theta_{target} \leftarrow` :guilabel:`polyak` :math:`\theta + (1 \;-` :guilabel:`polyak` :math:`) \theta_{target}`
|         :math:`{\phi 1}_{target} \leftarrow` :guilabel:`polyak` :math:`{\phi 1} + (1 \;-` :guilabel:`polyak` :math:`) {\phi 1}_{target}`
|         :math:`{\phi 2}_{target} \leftarrow` :guilabel:`polyak` :math:`{\phi 2} + (1 \;-` :guilabel:`polyak` :math:`) {\phi 2}_{target}`
|     :green:`# update learning rate`
|     **IF** there is a :guilabel:`learning_rate_scheduler` **THEN**
|         step :math:`\text{scheduler}_\theta (\text{optimizer}_\theta)`
|         step :math:`\text{scheduler}_\phi (\text{optimizer}_\phi)`

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: skrl.agents.torch.td3.td3.TD3_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/td3/td3.py
   :language: python
   :lines: 16-55
   :linenos:

Spaces and models
^^^^^^^^^^^^^^^^^

The implementation supports the following `Gym spaces <https://www.gymlibrary.dev/content/spaces>`_

.. list-table::
   :header-rows: 1

   * - Gym spaces
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

The implementation uses 6 deterministic function approximators. These function approximators (models) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`models`

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

API
^^^

.. autoclass:: skrl.agents.torch.td3.td3.TD3
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
