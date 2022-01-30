Deep Deterministic Policy Gradient (DDPG)
=========================================

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

**Decision making** (:literal:`act(...)`)

| :math:`a \leftarrow \mu_\theta(s)`
| :math:`noise \leftarrow OU(\theta, \mu, base\_scale)`
| :math:`scale \leftarrow (1 - \frac{\text{timestep}}{scale_{timesteps}}) \; (scale_{initial} - scale_{final}) + scale_{final}`
| :math:`a \leftarrow \text{clip}(a + noise * scale, {a}_{Low}, {a}_{High})`

**Learning algorithm** (:literal:`_update(...)`)

| :green:`# sample a batch from memory`
| :math:`s, a, r, s', d \leftarrow` states, actions, rewards, next_states, dones
| :green:`# gradient steps`
| **FOR** each gradient step **DO**
|     :green:`# compute target values`
|     :math:`a' \leftarrow \mu_{\theta_{target}}(s')`
|     :math:`Q_{_{target}} \leftarrow Q_{\phi_{target}}(s', a')`
|     :math:`y \leftarrow r + \gamma \; \neg d \; Q_{_{target}}`
|     :green:`# compute critic loss`
|     :math:`Q \leftarrow Q_\phi(s, a)`
|     :math:`{Loss}_{critic} \leftarrow \frac{1}{N} \sum_{i=1}^N (Q - y)^2`
|     :green:`# optimize critic`
|     :math:`\nabla_{\phi} {Loss}_{critic}`
|     :green:`# compute policy (actor) loss`
|     :math:`a \leftarrow \mu_\theta(s)`
|     :math:`Q \leftarrow Q_\phi(s, a)`
|     :math:`{Loss}_{policy} \leftarrow - \frac{1}{N} \sum_{i=1}^N Q`
|     :green:`# optimize policy (actor)`
|     :math:`\nabla_{\theta} {Loss}_{policy}`
|     :green:`# update target networks`
|     :math:`\theta_{target} \leftarrow \tau \; \theta + (1 - \tau) \theta_{target}`
|     :math:`\phi_{target} \leftarrow \tau \; \phi + (1 - \tau) \phi_{target}`

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: skrl.agents.torch.ddpg.ddpg.DDPG_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/ddpg/ddpg.py
   :language: python
   :lines: 15-43
   :linenos:

Models (networks)
^^^^^^^^^^^^^^^^^

The implementation uses 4 deterministic function approximators. These function approximators (models) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`networks`

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
   * - :math:`Q_\phi(s, a)`
     - Q-network (critic)
     - :literal:`"critic"`
     - :ref:`Deterministic <models_deterministic>`
   * - :math:`Q_{\phi_{target}}(s, a)`
     - Target Q-network
     - :literal:`"target_critic"`
     - :ref:`Deterministic <models_deterministic>`

API
^^^

.. autoclass:: skrl.agents.torch.ddpg.ddpg.DDPG
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
