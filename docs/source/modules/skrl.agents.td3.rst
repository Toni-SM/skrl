TD3
===

Twin Delayed TD3 (TD3)
----------------------

| **FOR** each gradient step **DO**
|     sample batch of transitions (states, actions, rewards, next_states, dones) from memory
|        :math:`s, a, r, s', d` 
|     target policy smoothing
|        :math:`a' = \mu_{\theta_{target}}(s')`
|        :math:`noises = a' + \text{clip}(\epsilon, -c, c)` for sampled noises (:math:`\epsilon`)
|        :math:`a' = a' + noises`
|        :math:`a' = \text{clip}(a', {a'}_{Low}, {a'}_{High})`
|     compute target values
|        :math:`Q_{1_{target}} = Q_{{\phi 1}_{target}}(s', a')`
|        :math:`Q_{2_{target}} = Q_{{\phi 2}_{target}}(s', a')`
|        :math:`Q_{_{target}} = \text{min}(Q_{1_{target}}, Q_{2_{target}})`
|        :math:`y = r + \gamma \; \neg d \; Q_{_{target}}`
|     compute critic loss
|        :math:`Q_1 = Q_{\phi 1}(s, a)`
|        :math:`Q_2 = Q_{\phi 2}(s, a)`
|        :math:`{Loss}_{critic} = \frac{1}{N} \sum_{i=1}^N (Q_1 - y)^2 + \frac{1}{N} \sum_{i=1}^N (Q_2 - y)^2`
|     optimize critic
|        :math:`\nabla_{\phi} {Loss}_{critic}`
|     **IF** it's time for the delayed update **THEN**
|        compute policy (actor) loss
|           :math:`a = \mu_\theta(s)`
|           :math:`Q_1 = Q_{\phi 1}(s, a)`
|           :math:`{Loss}_{policy} = - \frac{1}{N} \sum_{i=1}^N Q_1`
|        optimize policy (actor)
|           :math:`\nabla_{\theta} {Loss}_{policy}`
|        update target networks
|           :math:`\theta_{target} = \tau \; \theta + (1 - \tau) \theta_{target}`
|           :math:`{\phi 1}_{target} = \tau \; {\phi 1} + (1 - \tau) {\phi 1}_{target}`
|           :math:`{\phi 2}_{target} = \tau \; {\phi 2} + (1 - \tau) {\phi 2}_{target}`

API
^^^

.. py:data:: skrl.agents.td3.td3.TD3_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/td3/td3.py
   :language: python
   :lines: 15-46
   :linenos:

.. autoclass:: skrl.agents.td3.td3.TD3
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
