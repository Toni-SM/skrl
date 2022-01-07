DDPG
====

Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------

| :green:`# gradient steps`
| **FOR** each gradient step **DO**
|     :green:`# sample a batch from memory`
|     :math:`s, a, r, s', d \leftarrow` states, actions, rewards, next_states, dones
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

API
^^^

.. py:data:: skrl.agents.torch.ddpg.ddpg.DDPG_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/ddpg/ddpg.py
   :language: python
   :lines: 13-40
   :linenos:

.. autoclass:: skrl.agents.torch.ddpg.ddpg.DDPG
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
