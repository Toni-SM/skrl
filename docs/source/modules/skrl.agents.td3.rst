TD3
===

Twin Delayed TD3 (TD3)
----------------------

| :green:`# gradient steps`
| **FOR** each gradient step **DO**
|     :green:`# sample a batch from memory`
|     :math:`s, a, r, s', d \leftarrow` states, actions, rewards, next_states, dones 
|     :green:`# target policy smoothing`
|     :math:`a' \leftarrow \mu_{\theta_{target}}(s')`
|     :math:`noises \leftarrow a' + \text{clip}(\epsilon, -c, c)` for sampled noises (:math:`\epsilon`)
|     :math:`a' \leftarrow a' + noises`
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
   :lines: 15-47
   :linenos:

API
^^^

.. autoclass:: skrl.agents.torch.td3.td3.TD3
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
