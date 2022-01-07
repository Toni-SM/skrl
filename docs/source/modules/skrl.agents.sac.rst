SAC
===

Soft Actor-Critic (SAC)
-----------------------

| :green:`# gradient steps`
| **FOR** each gradient step **DO**
|     :green:`# sample a batch from memory`
|     :math:`s, a, r, s', d \leftarrow` states, actions, rewards, next_states, dones 
|     :green:`# compute target values`
|     :math:`a',\; logp' \leftarrow \pi_\theta(s')`
|     :math:`Q_{1_{target}} \leftarrow Q_{{\phi 1}_{target}}(s', a')`
|     :math:`Q_{2_{target}} \leftarrow Q_{{\phi 2}_{target}}(s', a')`
|     :math:`Q_{_{target}} \leftarrow \text{min}(Q_{1_{target}}, Q_{2_{target}}) - \alpha \; logp'`
|     :math:`y \leftarrow r + \gamma \; \neg d \; Q_{_{target}}`
|     :green:`# compute critic loss`
|     :math:`Q_1 \leftarrow Q_{\phi 1}(s, a)`
|     :math:`Q_2 \leftarrow Q_{\phi 2}(s, a)`
|     :math:`{Loss}_{critic} \leftarrow 0.5 \; (\frac{1}{N} \sum_{i=1}^N (Q_1 - y)^2 + \frac{1}{N} \sum_{i=1}^N (Q_2 - y)^2)`
|     :green:`# optimize critic`
|     :math:`\nabla_{\phi} {Loss}_{critic}`
|     :green:`# compute policy (actor) loss`
|     :math:`a,\; logp \leftarrow \pi_\theta(s)`
|     :math:`Q_1 \leftarrow Q_{\phi 1}(s, a)`
|     :math:`Q_2 \leftarrow Q_{\phi 2}(s, a)`
|     :math:`{Loss}_{policy} \leftarrow \frac{1}{N} \sum_{i=1}^N (\alpha \; logp - \text{min}(Q_1, Q_2))`
|     :green:`# optimize policy (actor)`
|     :math:`\nabla_{\theta} {Loss}_{policy}`
|     :green:`# entropy learning`
|     **IF** entropy learning is enabled **THEN**
|         :green:`# compute entropy loss`
|         :math:`{Loss}_{entropy} \leftarrow - \frac{1}{N} \sum_{i=1}^N (log(\alpha) \; (logp + \alpha_{Target}))`
|         :green:`# optimize entropy`
|         :math:`\nabla_{\alpha} {Loss}_{entropy}`
|         :green:`# compute entropy coefficient`
|         :math:`\alpha \leftarrow e^{log(\alpha)}`
|     :green:`# update target networks`
|     :math:`{\phi 1}_{target} \leftarrow \tau {\phi 1} + (1 - \tau) {\phi 1}_{target}`
|     :math:`{\phi 2}_{target} \leftarrow \tau {\phi 2} + (1 - \tau) {\phi 2}_{target}`

API
^^^

.. py:data:: skrl.agents.torch.sac.sac.SAC_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/sac/sac.py
   :language: python
   :lines: 16-41
   :linenos:

.. autoclass:: skrl.agents.torch.sac.sac.SAC
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
