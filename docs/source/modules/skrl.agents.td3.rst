Twin-Delayed DDPG (TD3)
=======================

TD3 is a **model-free**, **deterministic** **off-policy** **actor-critic** algorithm (based on DDPG) that relies on double Q-learning, target policy smoothing and delayed policy updates to address the problems introduced by overestimation bias in actor-critic algorithms 

Paper: `Addressing Function Approximation Error in Actor-Critic Methods <https://arxiv.org/abs/1802.09477>`_

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

**Decision making** (:literal:`act(...)`)

| :math:`a \leftarrow \mu_\theta(s)`
| :math:`noise \leftarrow N(\mu, \sigma)`
| :math:`scale \leftarrow (1 - \frac{\text{timestep}}{scale_{timesteps}}) \; (scale_{initial} - scale_{final}) + scale_{final}`
| :math:`a \leftarrow \text{clip}(a + noise * scale, {a}_{Low}, {a}_{High})`

**Learning algorithm** (:literal:`_update(...)`)

| :green:`# sample a batch from memory`
| :math:`s, a, r, s', d \leftarrow` states, actions, rewards, next_states, dones 
| :green:`# gradient steps`
| **FOR** each gradient step **DO**
|     :green:`# target policy smoothing`
|     :math:`a' \leftarrow \mu_{\theta_{target}}(s')`
|     :math:`noise \leftarrow \text{clip}(\epsilon, -c, c) \qquad` where :math:`\; \epsilon \leftarrow N(\mu, \sigma)`
|     :math:`a' \leftarrow a' + noise`
|     :math:`a' \leftarrow \text{clip}(a', {a'}_{Low}, {a'}_{High})`
|     :green:`# compute target values`
|     :math:`Q_{1_{target}} \leftarrow Q_{{\phi 1}_{target}}(s', a')`
|     :math:`Q_{2_{target}} \leftarrow Q_{{\phi 2}_{target}}(s', a')`
|     :math:`Q_{_{target}} \leftarrow \text{min}(Q_{1_{target}}, Q_{2_{target}})`
|     :math:`y \leftarrow r + \gamma \; \neg d \; Q_{_{target}}`
|     :green:`# compute critic loss`
|     :math:`Q_1 \leftarrow Q_{\phi 1}(s, a)`
|     :math:`Q_2 \leftarrow Q_{\phi 2}(s, a)`
|     :math:`{Loss}_{critic} \leftarrow \frac{1}{N} \sum_{i=1}^N (Q_1 - y)^2 + \frac{1}{N} \sum_{i=1}^N (Q_2 - y)^2`
|     :green:`# optimize critic`
|     :math:`\nabla_{\phi} {Loss}_{critic}`
|     :green:`# delayed update`
|     **IF** it's time for the delayed update **THEN**
|         :green:`# compute policy (actor) loss`
|         :math:`a \leftarrow \mu_\theta(s)`
|         :math:`Q_1 \leftarrow Q_{\phi 1}(s, a)`
|         :math:`{Loss}_{policy} \leftarrow - \frac{1}{N} \sum_{i=1}^N Q_1`
|         :green:`# optimize policy (actor)`
|         :math:`\nabla_{\theta} {Loss}_{policy}`
|         :green:`# update target networks`
|         :math:`\theta_{target} \leftarrow \tau \; \theta + (1 - \tau) \theta_{target}`
|         :math:`{\phi 1}_{target} \leftarrow \tau \; {\phi 1} + (1 - \tau) {\phi 1}_{target}`
|         :math:`{\phi 2}_{target} \leftarrow \tau \; {\phi 2} + (1 - \tau) {\phi 2}_{target}`

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: skrl.agents.torch.td3.td3.TD3_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/td3/td3.py
   :language: python
   :lines: 16-48
   :linenos:

Spaces and models
^^^^^^^^^^^^^^^^^

The implementation supports the following `Gym spaces <https://gym.openai.com/docs/#spaces>`_

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
     - Type
   * - :math:`\mu_\theta(s)`
     - Policy (actor)
     - :literal:`"policy"`
     - :ref:`Deterministic <models_deterministic>`
   * - :math:`\mu_{\theta_{target}}(s)`
     - Target policy
     - :literal:`"target_policy"`
     - :ref:`Deterministic <models_deterministic>`
   * - :math:`Q_{\phi 1}(s, a)`
     - Q1-network (critic 1)
     - :literal:`"critic_1"`
     - :ref:`Deterministic <models_deterministic>`
   * - :math:`Q_{\phi 2}(s, a)`
     - Q2-network (critic 2)
     - :literal:`"critic_2"`
     - :ref:`Deterministic <models_deterministic>`
   * - :math:`Q_{{\phi 1}_{target}}(s, a)`
     - Target Q1-network
     - :literal:`"target_critic_1"`
     - :ref:`Deterministic <models_deterministic>`
   * - :math:`Q_{{\phi 2}_{target}}(s, a)`
     - Target Q2-network
     - :literal:`"target_critic_2"`
     - :ref:`Deterministic <models_deterministic>`

API
^^^

.. autoclass:: skrl.agents.torch.td3.td3.TD3
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
