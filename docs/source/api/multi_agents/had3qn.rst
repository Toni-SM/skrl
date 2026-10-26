:tocdepth: 4

Heterogeneous-Agent Dueling Double Deep Q-Network (HAD3QN)
==========================================================

HAD3QN is a **model-free**, **off-policy** **value-based** CTDE (centralized training, decentralized execution)
**multi-agent** algorithm for cooperative tasks with **discrete** action spaces. Each agent has its own Q-network,
which takes the agent's observations and is used to act (decentralized execution). A centralized critic, shared by all
the agents, takes the environment states and estimates the value of every joint action of the agents. The agents'
Q-networks are updated **sequentially** (one agent after the other): each agent's Q-network is regressed towards the
critic's value of the joint action, in which the agents updated before it take the best actions according to the
critic.

**Paper:** `Heterogeneous-Agent Reinforcement Learning <https://jmlr.org/papers/v25/23-0488.html>`_.

|br| |hr|

Algorithm
---------

| For each gradient step do:
|     :math:`\bullet \;` Sample the same transitions (observations :math:`o`, states :math:`s`, actions :math:`a`, rewards :math:`r`, next observations :math:`o'`, next states :math:`s'` and terminated :math:`d_{_{end}}`) from the replay memories of all the agents
|     :math:`\bullet \;` Update the critic :math:`Q_\phi` towards the target computed with the next joint action made of the greedy actions of all the agents' target Q-networks :math:`a_k' = \arg\max_{a} Q_{\theta_{k, target}}(o_k', a)`
|     :math:`\bullet \;` Draw a permutation :math:`i_{1:n}` of the agents (or use the order of the possible agents)
|     :math:`\bullet \;` Start from the agents' greedy actions :math:`a_k = \arg\max_{a} Q_{\theta_k}(o_k, a)`
|     For each agent :math:`i_m` in order do:
|         :math:`\bullet \;` Regress the agent's Q-value of its action, :math:`Q_{\theta_{i_m}}(o_{i_m}, a_{i_m})`, towards the critic's value of the joint action :math:`Q_\phi(s, \boldsymbol{a})`
|         :math:`\bullet \;` Replace the agent's action by the best action according to the critic, the other agents' actions being fixed: :math:`a_{i_m} \leftarrow \arg\max_{a} Q_\phi(s, \boldsymbol{a}_{-i_m} \cup a)`
|     :math:`\bullet \;` Update the target networks every :guilabel:`target_update_interval` gradient steps

|

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - Q-network function approximator of the agent :math:`k` (:math:`Q_{\theta_k}`), critic function approximator (:math:`Q_\phi`)
|   - observations (:math:`o`), states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next observations (:math:`o'`), next states (:math:`s'`), terminated (:math:`d_{_{end}}`)
|   - number of actions of the agent :math:`k` (:math:`n_k`)
|   - joint action (:math:`\boldsymbol{a}`): the index of the agents' actions in the joint action space, computed as :math:`\text{joint}(a_1, \dots, a_n) = \sum_k a_k \prod_{j < k} n_j`
|   - loss (:math:`L`)

|

Decision making
"""""""""""""""

|
| :literal:`act(...)`
| **FOR** each agent :math:`k` **DO**
|     **IF** timestep :math:`<` :guilabel:`random_timesteps` **THEN**
|         **RETURN** random action :math:`a_k`
|     **IF** there is no :guilabel:`exploration_scheduler` **THEN**
|         **RETURN** :math:`a_k \leftarrow \underset{a}{\arg\max} \; Q_{\theta_k}(o_k, a)`
|     :math:`\epsilon \leftarrow` :guilabel:`exploration_scheduler` :math:`(\text{timestep}, \text{timesteps})`
|     :math:`a_k \leftarrow \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \underset{a}{\arg\max} \; Q_{\theta_k}(o_k, a) & \text{otherwise} \end{cases}`

|

Learning algorithm
""""""""""""""""""

|
| :literal:`update(...)`
| :green:`# gradient steps`
| **FOR** each gradient step up to :guilabel:`gradient_steps` **DO**
|     :green:`# sample the same transitions (same memory indexes) from the memories of all the agents`
|     [:math:`o, s, a, r, o', s', d_{_{end}}`] with size :guilabel:`batch_size`
|     :green:`# compute target values with the team reward (the mean of the agents' rewards) and the team episode end`
|     :green:`# (all the agents have terminated), and the target Q-networks' greedy next joint action`
|     :math:`\boldsymbol{a}' \leftarrow \text{joint}(\underset{a}{\arg\max} \; Q_{\theta_{1, target}}(o_1', a), \dots, \underset{a}{\arg\max} \; Q_{\theta_{n, target}}(o_n', a))`
|     :math:`Q_{_{target}} \leftarrow Q_{\phi_{target}}(s')[\boldsymbol{a}']`
|     :math:`y \leftarrow r \;+` :guilabel:`discount_factor` :math:`\neg d_{_{end}} \; Q_{_{target}}`
|     :green:`# compute critic loss`
|     :math:`Q \leftarrow Q_\phi(s)[\text{joint}(a_1, \dots, a_n)]`
|     :math:`L_{Q_\phi} \leftarrow \frac{1}{N} \sum_{i=1}^N (Q - y)^2`
|     :green:`# optimization step (critic)`
|     reset :math:`\text{optimizer}_\phi`
|     :math:`\nabla_{\phi} L_{Q_\phi}`
|     step :math:`\text{optimizer}_\phi`
|     :green:`# sequential Q-network updates, in a fixed (possible agents' order) or random order`
|     **IF** :guilabel:`fixed_order` **THEN**
|         :math:`i_{1:n} \leftarrow` possible agents' order
|     **ELSE**
|         :math:`i_{1:n} \leftarrow` random permutation of the agents
|     :green:`# start from the agents' greedy actions, and compute the critic's values of all the joint actions once`
|     :math:`a_k \leftarrow \underset{a}{\arg\max} \; Q_{\theta_k}(o_k, a)` for each agent :math:`k`
|     :math:`Q_{_{all}} \leftarrow Q_\phi(s)`
|     **FOR** each agent :math:`k` in :math:`i_{1:n}` **DO**
|         :green:`# compute Q-network loss`
|         :math:`L_{Q_{\theta_k}} \leftarrow \frac{1}{N} \sum_{i=1}^N (Q_{\theta_k}(o_k)[a_k] - Q_{_{all}}[\text{joint}(a_1, \dots, a_n)])^2`
|         :green:`# optimization step (Q-network)`
|         reset :math:`\text{optimizer}_{\theta_k}`
|         :math:`\nabla_{\theta_k} L_{Q_{\theta_k}}`
|         step :math:`\text{optimizer}_{\theta_k}`
|         :green:`# replace the agent's action by the best action according to the critic (the other agents' actions`
|         :green:`# being fixed), for the next agents`
|         :math:`a_k \leftarrow \underset{a}{\arg\max} \; Q_{_{all}}[\text{joint}(a_1, \dots, a_{k-1}, a, a_{k+1}, \dots, a_n)]`
|     :green:`# update target networks`
|     **IF** it's time to update the target networks (every :guilabel:`target_update_interval` gradient steps) **THEN**
|         **FOR** each agent :math:`k` **DO**
|             :math:`\theta_{k, target} \leftarrow` :guilabel:`polyak` :math:`\theta_k + (1 \;-` :guilabel:`polyak` :math:`) \theta_{k, target}`
|         :math:`\phi_{target} \leftarrow` :guilabel:`polyak` :math:`\phi + (1 \;-` :guilabel:`polyak` :math:`) \phi_{target}`
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
                    :start-after: [start-had3qn-torch]
                    :end-before: [end-had3qn-torch]

|

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1

    * - Dataclass
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - ``HAD3QN_CFG``
      - :py:class:`~skrl.multi_agents.torch.had3qn.HAD3QN_CFG`
      -
      -

.. note::

    :guilabel:`learning_rate` (as well as :guilabel:`learning_rate_scheduler` and
    :guilabel:`learning_rate_scheduler_kwargs`) is a pair of values: the first one for the agent's Q-network and the
    second one for the critic. If a single value is given, it is used for both networks. By default, the Q-networks'
    learning rate is ``5e-4`` and the critic's learning rate is ``1e-3``.

.. note::

    :guilabel:`fixed_order`, :guilabel:`gradient_steps`, :guilabel:`batch_size`, :guilabel:`random_timesteps`,
    :guilabel:`learning_starts`, :guilabel:`update_interval` and :guilabel:`target_update_interval` apply to all the
    agents at once (they cannot be specified per agent), since every gradient step updates all the agents.

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
      - .. centered:: :math:`\blacksquare`
    * - MultiDiscrete
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
    * - Box
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - Dict
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

The agents' action spaces must be :literal:`Discrete` spaces (with :literal:`start` equal to 0). Otherwise, a
:literal:`ValueError` is raised when the agent is instantiated.

|

Models
^^^^^^

The implementation uses 4 deterministic function approximators per agent (the critic and target critic are shared by
all the agents). These function approximators (models) must be collected in a dictionary and passed to the
constructor of the class under the argument :literal:`models`.

.. list-table::
    :header-rows: 1

    * - Notation
      - Concept
      - Key
      - Input shape
      - Output shape
      - Type
    * - :math:`Q_{\theta_k}(o)`
      - Q-network
      - :literal:`"q_network"`
      - observation
      - action (:math:`n_k`)
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_{\theta_{k, target}}(o)`
      - Target Q-network
      - :literal:`"target_q_network"`
      - observation
      - action (:math:`n_k`)
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_\phi(s)`
      - Critic (shared)
      - :literal:`"critic"`
      - state
      - joint action (:math:`\prod_k n_k`)
      - :ref:`Deterministic <models_deterministic>`
    * - :math:`Q_{\phi_{target}}(s)`
      - Target critic (shared)
      - :literal:`"target_critic"`
      - state
      - joint action (:math:`\prod_k n_k`)
      - :ref:`Deterministic <models_deterministic>`

The critic (and target critic) is centralized: it takes the environment state as :literal:`inputs["states"]` and
outputs a value for each joint action of all the agents. Therefore, it must be built with the joint action space of all
the agents, a :literal:`Discrete` space whose size is the product of the agents' numbers of actions, which is computed
from the agents' action spaces by :py:func:`~skrl.utils.spaces.torch.compute_joint_space`, e.g.:

.. code-block:: python

    from skrl.utils.spaces.torch import compute_joint_space

    joint_action_space = compute_joint_space(
        [env.action_space(uid) for uid in env.possible_agents], mode="product"
    )

A critic whose number of actions differs from the size of the joint action space raises a :literal:`ValueError` when
the agent is instantiated. The joint action index of the agents' individual actions is computed in little-endian
mixed radix, in the order of the possible agents (e.g., for 2 agents with 4 and 3 actions, the individual actions
:math:`(1, 2)` map to the joint action :math:`1 + 4 \cdot 2 = 9`).

.. warning::

    The size of the joint action space (the critic's output) grows **exponentially** with the number of agents
    (:math:`\prod_k n_k`). For example, 5 agents with 10 actions each require a critic with :math:`10^5` outputs.
    HAD3QN is therefore suited to tasks with few agents and small discrete action spaces.

.. note::

    The critic must be shared by all the agents: the same model instances (critic and target critic) are passed for
    every agent. Otherwise, a :literal:`ValueError` is raised when the agent is instantiated. The critic is trained
    once per gradient step on the team reward (the mean of the agents' rewards), where the episodes end when all the
    agents have terminated. The agents' state spaces, as well as the critic-related configuration
    (:guilabel:`discount_factor`, :guilabel:`polyak`, the critic's :guilabel:`learning_rate`,
    :guilabel:`learning_rate_scheduler` and :guilabel:`learning_rate_scheduler_kwargs`, :guilabel:`state_preprocessor`
    and :guilabel:`state_preprocessor_kwargs`), must be the same for all the agents.

.. note::

    The *dueling* architecture (separate state-value and advantage streams) is a choice of the Q-network
    and critic models. Any deterministic model that outputs a value per action can be used.

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

.. automodule:: skrl.multi_agents.torch.had3qn
.. autosummary::
    :nosignatures:

    HAD3QN_CFG
    HAD3QN

.. autoclass:: skrl.multi_agents.torch.had3qn.HAD3QN_CFG
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.multi_agents.torch.had3qn.HAD3QN
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
