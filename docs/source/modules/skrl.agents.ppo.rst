Proximal Policy Optimization (PPO)
==================================

PPO is a **model-free**, **stochastic** **on-policy** **policy gradient** algorithm that alternates between sampling data through interaction with the environment, and optimizing a *surrogate* objective function while avoiding that the new policy does not move too far away from the old one

Paper: `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

**Learning algorithm** (:literal:`_update(...)`)

| :green:`# compute returns and advantages`
| :math:`V \leftarrow V_\phi(s')`
| :green:`# sample mini-batches from memory`
| [[:math:`s, a, logp, V, R, A`]] :math:`\leftarrow` states, actions, log_prob, values, returns, advantages
| :green:`# learning epochs`
| **FOR** each learning epoch **DO**
|     :green:`# mini-batches loop`
|     **FOR** each mini-batch [:math:`s, a, logp, V, R, A`] in mini-batches **DO**
|          :math:`logp' \leftarrow \pi_\theta(s, a)`
|          :green:`# early stopping with KL divergence`
|          **IF** early stopping with KL divergence is enabled **THEN**
|              :math:`ratio \leftarrow logp' - logp`
|              :math:`KL_{_{divergence}} \leftarrow \frac{1}{N} \sum_{i=1}^N ((e^{ratio} - 1) - ratio)`
|              **IF** :math:`KL_{_{divergence}} > KL_{Threshold}` **THEN**
|                  **BREAK LOOP**
|          :green:`# compute entropy loss`
|          **IF** entropy computation is enabled **THEN**
|              :math:`{Loss}_{entropy} \leftarrow - entropy_{Loss \: scale} \; \frac{1}{N} \sum_{i=1}^N \pi_{\theta_{entropy}}`
|          **ELSE**
|              :math:`{Loss}_{entropy} \leftarrow 0`
|          :green:`# compute policy loss`
|          :math:`ratio \leftarrow e^{logp' - logp}`
|          :math:`surrogate \leftarrow A \; ratio`
|          :math:`surrogate_{clipped} \leftarrow A \; \text{clip}(ratio, 1 - c, 1 + c)`
|          :math:`{Loss}_{policy} \leftarrow - \frac{1}{N} \sum_{i=1}^N \text(min)(surrogate, surrogate_{clipped})`
|          :green:`# optimize policy`
|          :math:`\nabla_{\theta} ({Loss}_{policy} + {Loss}_{entropy})`
|          :green:`# compute value loss`
|          :math:`V' \leftarrow V_\phi(s')`
|          **IF** clipping predicted values is enabled **THEN**
|              :math:`V' \leftarrow V + \text{clip}(V' - V, -c, c)`
|          :math:`{Loss}_{value} \leftarrow value_{Loss \: scale} \; \frac{1}{N} \sum_{i=1}^N (R - V')^2`
|          :green:`# optimize value`
|          :math:`\nabla_{\phi} {Loss}_{value}`

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: skrl.agents.torch.ppo.ppo.PPO_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/ppo/ppo.py
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
     - .. centered:: :math:`\blacksquare`
   * - Box
     - .. centered:: :math:`\blacksquare`
     - .. centered:: :math:`\blacksquare`

The implementation uses 1 stochastic (discrete or continuous) and 1 deterministic function approximator. These function approximators (models) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`models`

.. list-table::
   :header-rows: 1

   * - Notation
     - Concept
     - Key
     - Type
   * - :math:`\pi_\theta(s)`
     - Policy
     - :literal:`"policy"`
     - :ref:`Categorical <models_categorical>` / :ref:`Gaussian <models_gaussian>`
   * - :math:`V_\phi(s)`
     - Value
     - :literal:`"value"`
     - :ref:`Deterministic <models_deterministic>`

API
^^^

.. autoclass:: skrl.agents.torch.ppo.ppo.PPO
   :undoc-members:
   :show-inheritance:
   :private-members: _update
   :members:
   
   .. automethod:: __init__
