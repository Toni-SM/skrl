DDPG
====

Deep Deterministic Policy Gradient (DDPG)
-----------------------------------------


| **FOR** each gradient step **DO**
|     sample batch of transitions (states, actions, rewards, next_states, dones) from memory
|        :math:`s, a, r, s', d`
|     compute target values
|        :math:`a' = \mu_{\theta_{target}}(s')`
|        :math:`Q_{_{target}} = Q_{\phi_{target}}(s', a')`
|        :math:`y = r + \gamma \; \neg d \; Q_{_{target}}`
|     compute critic loss
|        :math:`Q = Q_\phi(s, a)`
|        :math:`{Loss}_{critic} = \frac{1}{N} \sum_{i=1}^N (Q - y)^2`
|     optimize critic
|        :math:`\nabla_{\phi} {Loss}_{critic}`
|     compute policy (actor) loss
|        :math:`a = \mu_\theta(s)`
|        :math:`Q = Q_\phi(s, a)`
|        :math:`{Loss}_{policy} = - \frac{1}{N} \sum_{i=1}^N Q`
|     optimize policy (actor)
|        :math:`\nabla_{\theta} {Loss}_{policy}`
|     update target networks
|        :math:`\theta_{target} = \tau \; \theta + (1 - \tau) \theta_{target}`
|        :math:`\phi_{target} = \tau \; \phi + (1 - \tau) \phi_{target}`

API
^^^

.. py:data:: skrl.agents.ddpg.ddpg.DDPG_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/ddpg/ddpg.py
   :language: python
   :lines: 14-35
   :linenos:

.. autoclass:: skrl.agents.ddpg.ddpg.DDPG
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
