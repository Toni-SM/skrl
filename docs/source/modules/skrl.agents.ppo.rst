PPO
===

Proximal Policy Optimization (PPO)
----------------------------------

| :green:`# compute returns and advantages`
| :math:`V \leftarrow V_\phi(s')`
| :green:`# sample all data from memory`
| :math:`s, a, logp, V, R, A \leftarrow` states, actions, log_prob, values, returns, advantages
| :green:`# learning epochs`
| **FOR** each learning epoch **DO**
|     :math:`logp' \leftarrow \pi_\theta(s, a)`
|     :green:`# early stopping with KL divergence`
|     **IF** early stopping with KL divergence is enabled **THEN**
|         :math:`ratio \leftarrow logp' - logp`
|         :math:`KL_{_{divergence}} \leftarrow \frac{1}{N} \sum_{i=1}^N ((e^{ratio} - 1) - ratio)`
|         **IF** :math:`KL_{_{divergence}} > KL_{Threshold}` **THEN**
|             **BREAK LOOP**
|     :green:`# compute entropy loss`
|     **IF** entropy computation is enabled **THEN**
|         :math:`{Loss}_{entropy} \leftarrow - entropy_{Loss \: scale} \; \frac{1}{N} \sum_{i=1}^N \pi_{\theta_{entropy}}`
|     **ELSE**
|         :math:`{Loss}_{entropy} \leftarrow 0`
|     :green:`# compute policy loss`
|     :math:`ratio \leftarrow e^{logp' - logp}`
|     :math:`surrogate \leftarrow A \; ratio`
|     :math:`surrogate_{clipped} \leftarrow A \; \text{clip}(ratio, 1 - c, 1 + c)`
|     :math:`{Loss}_{policy} \leftarrow - \frac{1}{N} \sum_{i=1}^N \text(min)(surrogate, surrogate_{clipped})`
|     :green:`# optimize policy`
|     :math:`\nabla_{\theta} ({Loss}_{policy} + {Loss}_{entropy})`
|     :green:`# compute value loss`
|     :math:`V' \leftarrow V_\phi(s')`
|     **IF** clipping predicted values is enabled **THEN**
|         :math:`V' \leftarrow V + \text{clip}(V' - V, -c, c)`
|     :math:`{Loss}_{value} \leftarrow value_{Loss \: scale} \; \frac{1}{N} \sum_{i=1}^N (R - V')^2`
|     :green:`# optimize value`
|     :math:`\nabla_{\phi} {Loss}_{value}`

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: skrl.agents.torch.ppo.ppo.PPO_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/ppo/ppo.py
   :language: python
   :lines: 14-44
   :linenos:

API
^^^

.. autoclass:: skrl.agents.torch.ppo.ppo.PPO
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
