SAC
===

Soft Actor-Critic (SAC)
-----------------------

| **FOR** each gradient step **DO**
|     sample batch of transitions (states, actions, rewards, next_states, dones) from memory
|        :math:`s, a, r, s', d` 
|     compute target values
|        :math:`a',\; logp' = \pi_\theta(s')`
|        :math:`Q_{1_{target}} = Q_{{\phi 1}_{target}}(s', a')`
|        :math:`Q_{2_{target}} = Q_{{\phi 2}_{target}}(s', a')`
|        :math:`Q_{_{target}} = \text{min}(Q_{1_{target}}, Q_{2_{target}}) - \alpha \; logp'`
|        :math:`y = r + \gamma \; \neg d \; Q_{_{target}}`
|     compute critic loss
|        :math:`Q_1 = Q_{\phi 1}(s, a)`
|        :math:`Q_2 = Q_{\phi 2}(s, a)`
|        :math:`{Loss}_{critic} = \frac{1}{N} \sum_{i=1}^N (Q_1 - y)^2 + \frac{1}{N} \sum_{i=1}^N (Q_2 - y)^2`
|     optimize critic
|        :math:`\nabla_{\phi} {Loss}_{critic}`
|     compute policy (actor) loss
|        :math:`a,\; logp = \pi_\theta(s)`
|        :math:`Q_1 = Q_{\phi 1}(s, a)`
|        :math:`Q_2 = Q_{\phi 2}(s, a)`
|        :math:`{Loss}_{policy} = \frac{1}{N} \sum_{i=1}^N (\alpha \; logp - \text{min}(Q_1, Q_2))`
|     optimize policy (actor)
|        :math:`\nabla_{\theta} {Loss}_{policy}`
|     **IF** entropy learning is enabled **THEN**
|        compute entropy loss
|           :math:`{Loss}_{entropy} = - \frac{1}{N} \sum_{i=1}^N (log(\alpha) \; (logp + \alpha_{Target}))`
|        optimize entropy
|           :math:`\nabla_{\alpha} {Loss}_{entropy}`
|        compute entropy coefficient
|           :math:`\alpha = e^{log(\alpha)}`
|     update target networks
|        :math:`{\phi 1}_{target} = \tau {\phi 1} + (1 - \tau) {\phi 1}_{target}`
|        :math:`{\phi 2}_{target} = \tau {\phi 2} + (1 - \tau) {\phi 2}_{target}`

.. py:data:: skrl.agents.sac.sac.SAC_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/sac/sac.py
   :language: python
   :lines: 16-35
   :linenos:

.. autoclass:: skrl.agents.sac.sac.SAC
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
