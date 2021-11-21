PPO
===

Proximal Policy Optimization (PPO)
----------------------------------

| compute returns and advantages
|     :math:`V = V_\phi(s')`
| sample all data (states, actions, log_prob, values, returns, advantages) from memory
|     :math:`s, a, logp, V, R, A`    
| **FOR** each learning epoch **DO**
|     :math:`logp' = \pi_\theta(s, a)`
|     **IF** early stopping with KL divergence is enabled **THEN**
|        :math:`ratio = logp' - logp`
|        :math:`KL_{_{divergence}} = \frac{1}{N} \sum_{i=1}^N ((e^{ratio} - 1) - ratio)`
|        **IF** :math:`KL_{_{divergence}} > KL_{Threshold}` **THEN**
|           **BREAK LOOP**
|     **IF** entropy computation is enabled **THEN**
|         :math:`{Loss}_{entropy} = - entropy_{Loss \: scale} \; \frac{1}{N} \sum_{i=1}^N \pi_{\theta_{entropy}}`
|     **ELSE**
|         :math:`{Loss}_{entropy} = 0`
|     compute policy loss
|        :math:`ratio = e^{logp' - logp}`
|        :math:`surrogate = A \; ratio`
|        :math:`surrogate_{clipped} = A \; \text{clip}(ratio, 1 - c, 1 + c)`
|        :math:`{Loss}_{policy} = - \frac{1}{N} \sum_{i=1}^N \text(min)(surrogate, surrogate_{clipped})`
|     optimize policy
|        :math:`\nabla_{\theta} ({Loss}_{policy} + {Loss}_{entropy})`
|     compute value loss
|        :math:`V' = V_\phi(s')`
|        **IF** clipping predicted values is enabled **THEN**
|           :math:`V' = V + \text{clip}(V' - V, -c, c)`
|        :math:`{Loss}_{value} = value_{Loss \: scale} \; \frac{1}{N} \sum_{i=1}^N (R - V')^2`
|     optimize value
|        :math:`\nabla_{\phi} {Loss}_{value}`

.. py:data:: skrl.agents.ppo.ppo.PPO_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/ppo/ppo.py
   :language: python
   :lines: 14-37
   :linenos:

.. autoclass:: skrl.agents.ppo.ppo.PPO
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
