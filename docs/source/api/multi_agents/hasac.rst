:tocdepth: 4

Heterogeneous-Agent Soft Actor-Critic (HASAC)
=============================================

HASAC is a **model-free**, **stochastic** **off-policy** **actor-critic** CTDE (centralized training, decentralized
execution) **multi-agent** algorithm for cooperative tasks with **continuous** action spaces.
It extends SAC (maximum entropy reinforcement learning) to multiple agents: twin centralized critics estimate the soft
value of the joint action of all the agents (the minimum of their target values is used to address overestimation
bias), and each agent has a stochastic policy regularized by its own entropy coefficient.
The agents' policies are updated sequentially (one agent after the other, in a random order). Each policy update
takes into account the actions sampled from the updated policies of the previously updated agents, without requiring
the agents to share parameters (agents can be heterogeneous).

**Paper:** `Maximum Entropy Heterogeneous-Agent Reinforcement Learning <https://openreview.net/forum?id=tmqOhBC4a5>`_.

|br| |hr|

Algorithm
---------

| For each gradient step do:
|     :math:`\bullet \;` Sample the same transitions (observations :math:`o`, states :math:`s`, actions :math:`a`, rewards :math:`r`, next observations :math:`o'`, next states :math:`s'` and terminated :math:`d_{_{end}}`) from the replay memories of all the agents
|     :math:`\bullet \;` Update the twin critics :math:`Q_{\phi 1}, Q_{\phi 2}` towards the minimum of the twin target critics evaluated at the next joint action, sampled from all the agents' policies :math:`\boldsymbol{a}' = a_1' \oplus \dots \oplus a_n', \; a_k' \sim \pi_{\theta_k}(o_k')`, minus the entropy term of all the agents :math:`\sum_k \alpha_k \; log \, \pi_{\theta_k}(a_k' | o_k')`
|     :math:`\bullet \;` Initialize the joint action with actions sampled from the current policies :math:`\boldsymbol{a} \leftarrow a_1 \oplus \dots \oplus a_n, \; a_k \sim \pi_{\theta_k}(o_k)`
|     For each agent :math:`k`, in a random order, do:
|         :math:`\bullet \;` Optimize the policy loss :math:`L_{\pi_{\theta_k}} = \mathbb{E}[\alpha_k \; log \, \pi_{\theta_k}(\boldsymbol{a}_k | o_k) - \text{min}(Q_{\phi 1}, Q_{\phi 2})(s, \boldsymbol{a})]`, where only the agent's own action :math:`\boldsymbol{a}_k \sim \pi_{\theta_k}(o_k)` is differentiable
|         :math:`\bullet \;` Optimize the agent's entropy coefficient :math:`\alpha_k` (if it is learned)
|         :math:`\bullet \;` Replace the agent's action in the joint action with an action sampled from its updated policy: :math:`\boldsymbol{a}_k \sim \pi_{\theta_k'}(o_k)`
|     :math:`\bullet \;` Update the target critics

|

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - policy function approximator of the agent :math:`k` (:math:`\pi_{\theta_k}`), twin critic function approximators (:math:`Q_{\phi 1}`, :math:`Q_{\phi 2}`)
|   - observations (:math:`o`), states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next observations (:math:`o'`), next states (:math:`s'`), terminated (:math:`d_{_{end}}`)
|   - joint action (:math:`\boldsymbol{a}`): the agents' actions concatenated (:math:`\oplus`) in the order of the possible agents, and the action of the agent :math:`k` in it (:math:`\boldsymbol{a}_k`)
|   - log probabilities (:math:`logp`), entropy coefficient of the agent :math:`k` (:math:`\alpha_k`)
|   - loss (:math:`L`)

|

Decision making
"""""""""""""""

|
| :literal:`act(...)`
| **FOR** each agent :math:`k` **DO**
|     :math:`a_k \sim \pi_{\theta_k}(o_k)`

|

Learning algorithm
""""""""""""""""""

|
| :literal:`update(...)`
| :green:`# gradient steps`
| **FOR** each gradient step up to :guilabel:`gradient_steps` **DO**
|     :green:`# sample the same transitions (same memory indexes) from the memories of all the agents`
|     [:math:`o, s, a, r, o', s', d_{_{end}}`] with size :guilabel:`batch_size`
|     :green:`# compute the joint actions`
|     :math:`\boldsymbol{a} \leftarrow a_1 \oplus \dots \oplus a_n`
|     :green:`# next joint action (sampled from the policies of all the agents) and entropy term`
|     **FOR** each agent :math:`k` **DO**
|         :math:`a_k',\; logp_k' \leftarrow \pi_{\theta_k}(o_k')`
|     :math:`\boldsymbol{a}' \leftarrow a_1' \oplus \dots \oplus a_n'`
|     :green:`# compute target values (for a shared critic, with the team reward: the mean of the agents' rewards,`
|     :green:`# and the team episode end: all the agents have terminated)`
|     :math:`Q_{1_{target}} \leftarrow Q_{{\phi 1}_{target}}(s', \boldsymbol{a}')`
|     :math:`Q_{2_{target}} \leftarrow Q_{{\phi 2}_{target}}(s', \boldsymbol{a}')`
|     :math:`Q_{_{target}} \leftarrow \text{min}(Q_{1_{target}}, Q_{2_{target}}) - \sum_k \alpha_k \; logp_k'`
|     :math:`y \leftarrow r \;+` :guilabel:`discount_factor` :math:`\neg d_{_{end}} \; Q_{_{target}}`
|     :green:`# compute critic loss`
|     :math:`Q_1 \leftarrow Q_{\phi 1}(s, \boldsymbol{a})`
|     :math:`Q_2 \leftarrow Q_{\phi 2}(s, \boldsymbol{a})`
|     :math:`L_{Q_\phi} \leftarrow 0.5 \; (\frac{1}{N} \sum_{i=1}^N (Q_1 - y)^2 + \frac{1}{N} \sum_{i=1}^N (Q_2 - y)^2)`
|     :green:`# optimization step (critics)`
|     reset :math:`\text{optimizer}_\phi`
|     :math:`\nabla_{\phi} L_{Q_\phi}`
|     :math:`\text{clip}(\lVert \nabla_{\phi} \rVert)` with :guilabel:`grad_norm_clip`
|     step :math:`\text{optimizer}_\phi`
|     :green:`# sequential policy updates, starting from actions sampled from the current policies`
|     :math:`\boldsymbol{a} \leftarrow a_1 \oplus \dots \oplus a_n, \; a_k \sim \pi_{\theta_k}(o_k)`
|     **FOR** each agent :math:`k` in a random permutation of the agents (or in order, if :guilabel:`fixed_order` is enabled) **DO**
|         :green:`# compute policy (actor) loss (only the agent's own action is differentiable)`
|         :math:`\boldsymbol{a}_k,\; logp_k \leftarrow \pi_{\theta_k}(o_k)`
|         :math:`Q_1 \leftarrow Q_{\phi 1}(s, \boldsymbol{a})`
|         :math:`Q_2 \leftarrow Q_{\phi 2}(s, \boldsymbol{a})`
|         :math:`L_{\pi_{\theta_k}} \leftarrow \frac{1}{N} \sum_{i=1}^N (\alpha_k \; logp_k - \text{min}(Q_1, Q_2))`
|         :green:`# optimization step (policy)`
|         reset :math:`\text{optimizer}_{\theta_k}`
|         :math:`\nabla_{\theta_k} L_{\pi_{\theta_k}}`
|         :math:`\text{clip}(\lVert \nabla_{\theta_k} \rVert)` with :guilabel:`grad_norm_clip`
|         step :math:`\text{optimizer}_{\theta_k}`
|         :green:`# entropy learning (agent's entropy coefficient)`
|         **IF** :guilabel:`learn_entropy` is enabled **THEN**
|             :green:`# compute entropy loss`
|             :math:`{L}_{entropy_k} \leftarrow - \frac{1}{N} \sum_{i=1}^N (log(\alpha_k) \; (logp_k + \alpha_{k, Target}))`
|             :green:`# optimization step (entropy)`
|             reset :math:`\text{optimizer}_{\alpha_k}`
|             :math:`\nabla_{\alpha_k} {L}_{entropy_k}`
|             step :math:`\text{optimizer}_{\alpha_k}`
|             :green:`# compute entropy coefficient`
|             :math:`\alpha_k \leftarrow e^{log(\alpha_k)}`
|         :green:`# replace the agent's action with an action sampled from its updated policy`
|         :math:`\boldsymbol{a}_k \sim \pi_{\theta_k}(o_k)`
|     :green:`# update target networks`
|     :math:`{\phi 1}_{target} \leftarrow` :guilabel:`polyak` :math:`{\phi 1} + (1 \;-` :guilabel:`polyak` :math:`) {\phi 1}_{target}`
|     :math:`{\phi 2}_{target} \leftarrow` :guilabel:`polyak` :math:`{\phi 2} + (1 \;-` :guilabel:`polyak` :math:`) {\phi 2}_{target}`
|     :green:`# update learning rate`
|     **IF** there is a :guilabel:`learning_rate_scheduler` **THEN**
|         step :math:`\text{scheduler}_{\theta_k} (\text{optimizer}_{\theta_k})` for each agent :math:`k`
|         step :math:`\text{scheduler}_\phi (\text{optimizer}_\phi)`

|

Usage
-----

.. tabs::

    .. tab:: Standard implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/multi_agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [start-hasac-torch]
                    :end-before: [end-hasac-torch]

|

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1

    * - Dataclass
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - ``HASAC_CFG``
      - :py:class:`~skrl.multi_agents.torch.hasac.HASAC_CFG`
      -
      -

.. note::

    :guilabel:`gradient_steps`, :guilabel:`batch_size`, :guilabel:`random_timesteps`, :guilabel:`learning_starts` and
    :guilabel:`fixed_order` apply to all the agents at once (they cannot be specified per agent), since every gradient
    step updates all the agents. The entropy settings (:guilabel:`learn_entropy`, :guilabel:`initial_entropy_value`,
    :guilabel:`target_entropy` and the entropy coefficient's :guilabel:`learning_rate`) are specified per agent: each
    agent has its own entropy coefficient. If not specified, an agent's :guilabel:`target_entropy` is the negative
    number of its actions.

|

Spaces
^^^^^^

The implementation supports the following `Gymnasium spaces <https://gymnasium.farama.org/api/spaces>`_:

.. list-table::
    :header-rows: 1

    * - Gymnasium spaces
      - .. centered:: Observation
      - .. centered:: Action
    * - Discrete
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - MultiDiscrete
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - Box
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - Dict
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

|

Models
^^^^^^

The implementation uses 1 stochastic and 4 deterministic function approximators per agent.
These function approximators (models) must be collected in a dictionary and passed to the constructor of the class
under the argument :literal:`models`.

.. list-table::
    :header-rows: 1

    * - Notation
      - Concept
      - Key
      - Input shape
      - Output shape
      - Type
    * - :math:`\pi_{\theta_k}(o)`
      - Policy (actor)
      - :literal:`"policy"`
      - observation
      - action
      - :ref:`Gaussian <models_gaussian>` /
        |br| :ref:`MultivariateGaussian <models_multivariate_gaussian>`
    * - :math:`Q_{\phi 1}(s, \boldsymbol{a})`
      - Q1-network (critic 1)
      - :literal:`"critic_1"`
      - state + joint action
      - 1
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_{\phi 2}(s, \boldsymbol{a})`
      - Q2-network (critic 2)
      - :literal:`"critic_2"`
      - state + joint action
      - 1
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_{{\phi 1}_{target}}(s, \boldsymbol{a})`
      - Target Q1-network
      - :literal:`"target_critic_1"`
      - state + joint action
      - 1
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_{{\phi 2}_{target}}(s, \boldsymbol{a})`
      - Target Q2-network
      - :literal:`"target_critic_2"`
      - state + joint action
      - 1
      - :ref:`Deterministic <models_deterministic>`

The critics (and target critics) are centralized: they take the joint action of all the agents (the agents' actions
concatenated in the order of the possible agents) as :literal:`inputs["taken_actions"]`, and the environment state as
:literal:`inputs["states"]`. Therefore, they must be built with the joint action space of all the agents, which is
computed from the agents' action spaces by :py:func:`~skrl.utils.spaces.torch.compute_joint_space`, e.g.:

.. code-block:: python

    from skrl.utils.spaces.torch import compute_joint_space

    joint_action_space = compute_joint_space(
        [env.action_space(uid) for uid in env.possible_agents], mode="concatenate"
    )

.. warning::

    The policies' log probabilities must be sum-reduced per sample (shape ``(N, 1)``), e.g.: the
    :ref:`Gaussian <models_gaussian>` model's ``reduction="sum"`` (default). Otherwise, the entropy terms of the
    target value and of the policy and entropy losses are wrong.

A critic whose number of actions differs from the size of the joint action space raises a :literal:`ValueError` when
the agent is instantiated.

.. note::

    The critics can be shared by all the agents or be specific to each agent. All the critic roles
    (:literal:`"critic_1"`, :literal:`"critic_2"`, :literal:`"target_critic_1"` and :literal:`"target_critic_2"`)
    must follow the same mode:

    * **Shared critics**: the same model instances (critics and target critics) are passed for every agent. The critics
      take the states and the joint action, and they are trained once per gradient step on the team reward (the mean
      of the agents' rewards), where the episodes end when all the agents have terminated. The agents' state spaces, as
      well as the critic-related configuration (:guilabel:`discount_factor`, :guilabel:`polyak`,
      :guilabel:`grad_norm_clip`, the critics' :guilabel:`learning_rate`, :guilabel:`learning_rate_scheduler` and
      :guilabel:`learning_rate_scheduler_kwargs`, :guilabel:`state_preprocessor` and
      :guilabel:`state_preprocessor_kwargs`), must be the same for all the agents.
    * **Per-agent critics**: distinct model instances are passed for every agent. Each pair of critics also takes its
      agent's own observations (:literal:`inputs["observations"]`), and it is trained on its agent's reward.

    In both modes, the twin critics of each group are optimized together by a single optimizer, and the target values
    subtract the entropy term of all the agents (since the critics evaluate the joint action).

.. note::

    The same transitions (memory indexes) are sampled from the memories of all the agents, so all the memories must
    have the same :literal:`memory_size` and :literal:`num_envs`. Otherwise, a :literal:`ValueError` is raised when
    the agent is instantiated.

|

Features
^^^^^^^^

Support for advanced features is described in the following table:

.. list-table::
    :header-rows: 1

    * - Feature
      - Support and remarks
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - Shared model
      - Critics shared by all the agents
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - RNN support
      - \-
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - Mixed precision
      - Automatic mixed precision
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - Distributed
      - Single Program Multi Data (SPMD) multi-GPU
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`

|

API
---

|

PyTorch
^^^^^^^

.. automodule:: skrl.multi_agents.torch.hasac
.. autosummary::
    :nosignatures:

    HASAC_CFG
    HASAC

.. autoclass:: skrl.multi_agents.torch.hasac.HASAC_CFG
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.multi_agents.torch.hasac.HASAC
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
