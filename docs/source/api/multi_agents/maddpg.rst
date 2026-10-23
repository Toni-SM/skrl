:tocdepth: 4

Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
=======================================================

MADDPG is a **model-free**, **deterministic** **off-policy** **actor-critic** CTDE (centralized training,
decentralized execution) **multi-agent** algorithm for cooperative tasks with **continuous** action spaces.
A centralized critic estimates the value of the joint action of all the agents, and each agent's policy is updated
by following the gradient of the critic with respect to its own action. Unlike
:doc:`HADDPG <haddpg>`, whose policies are updated sequentially (each update takes into account the updated actions
of the previously updated agents), the policies are updated simultaneously: every update takes into account the
**sampled** (replay buffer) actions of the other agents, so the result does not depend on the order in which
the agents are updated.

**Paper:** `Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments <https://arxiv.org/abs/1706.02275>`_.

|br| |hr|

Algorithm
---------

| For each gradient step do:
|     :math:`\bullet \;` Sample the same transitions (observations :math:`o`, states :math:`s`, actions :math:`a`, rewards :math:`r`, next observations :math:`o'`, next states :math:`s'` and terminated :math:`d_{_{end}}`) from the replay memories of all the agents
|     :math:`\bullet \;` Update the critic :math:`Q_\phi` towards the target computed with the next joint action of all the agents' target policies :math:`\boldsymbol{a}' = \mu_{\theta_{1, target}}(o_1') \oplus \dots \oplus \mu_{\theta_{n, target}}(o_n')`
|     For each agent :math:`k` do:
|         :math:`\bullet \;` Replace the agent's action in the sampled joint action with the action of its current policy: :math:`\boldsymbol{a}_k \leftarrow \mu_{\theta_k}(o_k)`, keeping the sampled actions of the other agents
|         :math:`\bullet \;` Optimize the policy loss :math:`L_{\mu_{\theta_k}} = -\mathbb{E}[Q_\phi(s, \boldsymbol{a})]`, where only the agent's own action :math:`\boldsymbol{a}_k` is differentiable
|     :math:`\bullet \;` Update the target networks

|

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - policy function approximator of the agent :math:`k` (:math:`\mu_{\theta_k}`), critic function approximator (:math:`Q_\phi`)
|   - observations (:math:`o`), states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next observations (:math:`o'`), next states (:math:`s'`), terminated (:math:`d_{_{end}}`)
|   - joint action (:math:`\boldsymbol{a}`): the agents' actions concatenated (:math:`\oplus`) in the order of the possible agents, and the action of the agent :math:`k` in it (:math:`\boldsymbol{a}_k`). The sampled joint action is the one stored in the replay memories
|   - loss (:math:`L`)

|

Decision making
"""""""""""""""

|
| :literal:`act(...)`
| **FOR** each agent :math:`k` **DO**
|     :math:`a_k \leftarrow \mu_{\theta_k}(o_k)`
|     :math:`noise \leftarrow` sample :guilabel:`exploration_noise`
|     **IF** there is an :guilabel:`exploration_scheduler` **THEN**
|         :math:`noise \leftarrow noise \;` :guilabel:`exploration_scheduler` :math:`(\text{timestep}, \text{timesteps})`
|     :math:`a_k \leftarrow \text{clip}(a_k + noise, {a_k}_{Low}, {a_k}_{High})`

|

Learning algorithm
""""""""""""""""""

|
| :literal:`update(...)`
| :green:`# gradient steps`
| **FOR** each gradient step up to :guilabel:`gradient_steps` **DO**
|     :green:`# sample the same transitions (same memory indexes) from the memories of all the agents`
|     [:math:`o, s, a, r, o', s', d_{_{end}}`] with size :guilabel:`batch_size`
|     :green:`# compute the joint actions (the next one, with the target policies of all the agents)`
|     :math:`\boldsymbol{a} \leftarrow a_1 \oplus \dots \oplus a_n`
|     :math:`\boldsymbol{a}' \leftarrow \mu_{\theta_{1, target}}(o_1') \oplus \dots \oplus \mu_{\theta_{n, target}}(o_n')`
|     :green:`# compute target values (for a shared critic, with the team reward: the mean of the agents' rewards,`
|     :green:`# and the team episode end: all the agents have terminated)`
|     :math:`Q_{_{target}} \leftarrow Q_{\phi_{target}}(s', \boldsymbol{a}')`
|     :math:`y \leftarrow r \;+` :guilabel:`discount_factor` :math:`\neg d_{_{end}} \; Q_{_{target}}`
|     :green:`# compute critic loss`
|     :math:`Q \leftarrow Q_\phi(s, \boldsymbol{a})`
|     :math:`L_{Q_\phi} \leftarrow \frac{1}{N} \sum_{i=1}^N (Q - y)^2`
|     :green:`# optimization step (critic)`
|     reset :math:`\text{optimizer}_\phi`
|     :math:`\nabla_{\phi} L_{Q_\phi}`
|     :math:`\text{clip}(\lVert \nabla_{\phi} \rVert)` with :guilabel:`grad_norm_clip`
|     step :math:`\text{optimizer}_\phi`
|     :green:`# policy updates (the order is irrelevant: the other agents' actions are always the sampled ones)`
|     **FOR** each agent :math:`k` **DO**
|         :green:`# compute policy (actor) loss: in the sampled joint action, only the agent's own action is replaced`
|         :green:`# by the action of its current policy (the other agents' actions are never updated)`
|         :math:`\boldsymbol{a}_k \leftarrow \mu_{\theta_k}(o_k)`
|         :math:`Q \leftarrow Q_\phi(s, \boldsymbol{a})`
|         :math:`L_{\mu_{\theta_k}} \leftarrow - \frac{1}{N} \sum_{i=1}^N Q`
|         :green:`# optimization step (policy)`
|         reset :math:`\text{optimizer}_{\theta_k}`
|         :math:`\nabla_{\theta_k} L_{\mu_{\theta_k}}`
|         :math:`\text{clip}(\lVert \nabla_{\theta_k} \rVert)` with :guilabel:`grad_norm_clip`
|         step :math:`\text{optimizer}_{\theta_k}`
|     :green:`# update target networks`
|     **FOR** each agent :math:`k` **DO**
|         :math:`\theta_{k, target} \leftarrow` :guilabel:`polyak` :math:`\theta_k + (1 \;-` :guilabel:`polyak` :math:`) \theta_{k, target}`
|     :math:`\phi_{target} \leftarrow` :guilabel:`polyak` :math:`\phi + (1 \;-` :guilabel:`polyak` :math:`) \phi_{target}`
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
                    :start-after: [start-maddpg-torch]
                    :end-before: [end-maddpg-torch]

|

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1

    * - Dataclass
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - ``MADDPG_CFG``
      - :py:class:`~skrl.multi_agents.torch.maddpg.MADDPG_CFG`
      -
      -

.. note::

    :guilabel:`gradient_steps`, :guilabel:`batch_size`, :guilabel:`random_timesteps` and :guilabel:`learning_starts`
    apply to all the agents at once (they cannot be specified per agent), since every gradient step updates all
    the agents.

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

The implementation uses 4 deterministic function approximators per agent.
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
    * - :math:`\mu_{\theta_k}(o)`
      - Policy (actor)
      - :literal:`"policy"`
      - observation
      - action
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`\mu_{\theta_{k, target}}(o)`
      - Target policy
      - :literal:`"target_policy"`
      - observation
      - action
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_\phi(s, \boldsymbol{a})`
      - Q-network (critic)
      - :literal:`"critic"`
      - state + joint action
      - 1
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_{\phi_{target}}(s, \boldsymbol{a})`
      - Target Q-network
      - :literal:`"target_critic"`
      - state + joint action
      - 1
      - :ref:`Deterministic <models_deterministic>`

The critic (and target critic) is centralized: it takes the joint action of all the agents (the agents' actions
concatenated in the order of the possible agents) as :literal:`inputs["taken_actions"]`, and the environment state as
:literal:`inputs["states"]`. Therefore, it must be built with the joint action space of all the agents, which is
computed from the agents' action spaces by :py:func:`~skrl.utils.spaces.torch.compute_joint_space`, e.g.:

.. code-block:: python

    from skrl.utils.spaces.torch import compute_joint_space

    joint_action_space = compute_joint_space(
        [env.action_space(uid) for uid in env.possible_agents], mode="concatenate"
    )

A critic whose number of actions differs from the size of the joint action space raises a :literal:`ValueError` when
the agent is instantiated.

.. note::

    The critic can be shared by all the agents or be specific to each agent:

    * **Shared critic**: the same model instances (critic and target critic) are passed for every agent. The critic
      takes the states and the joint action, and it is trained once per gradient step on the team reward (the mean of
      the agents' rewards), where the episodes end when all the agents have terminated. The agents' state spaces, as
      well as the critic-related configuration (:guilabel:`discount_factor`, :guilabel:`polyak`,
      :guilabel:`grad_norm_clip`, the critic's :guilabel:`learning_rate`, :guilabel:`learning_rate_scheduler` and
      :guilabel:`learning_rate_scheduler_kwargs`, :guilabel:`state_preprocessor` and
      :guilabel:`state_preprocessor_kwargs`), must be the same for all the agents.
    * **Per-agent critic**: distinct model instances are passed for every agent. Each critic also takes its agent's
      own observations (:literal:`inputs["observations"]`), and it is trained on its agent's reward.

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
      - Critic shared by all the agents
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

.. automodule:: skrl.multi_agents.torch.maddpg
.. autosummary::
    :nosignatures:

    MADDPG_CFG
    MADDPG

.. autoclass:: skrl.multi_agents.torch.maddpg.MADDPG_CFG
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.multi_agents.torch.maddpg.MADDPG
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
