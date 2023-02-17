Cross-Entropy Method (CEM)
==========================

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - policy function approximator (:math:`\pi_\theta`)
|   - states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next states (:math:`s'`), dones (:math:`d`)
|   - loss (:math:`L`)

**Decision making** (:literal:`act(...)`)

| :math:`a \leftarrow \pi_\theta(s)`

**Learning algorithm** (:literal:`_update(...)`)

| :green:`# sample all memory`
| :math:`s, a, r, s', d \leftarrow` states, actions, rewards, next_states, dones
| :green:`# compute discounted return threshold`
| :math:`[G] \leftarrow \sum_{t=0}^{E-1}` :guilabel:`discount_factor`:math:`^{t} \, r_t` for each episode
| :math:`G_{_{bound}} \leftarrow q_{th_{quantile}}([G])` at the given :guilabel:`percentile`
| :green:`# get elite states and actions`
| :math:`s_{_{elite}} \leftarrow s[G \geq G_{_{bound}}]`
| :math:`a_{_{elite}} \leftarrow a[G \geq G_{_{bound}}]`
| :green:`# compute scores for the elite states`
| :math:`scores \leftarrow \theta(s_{_{elite}})`
| :green:`# compute policy loss`
| :math:`L_{\pi_\theta} \leftarrow -\sum_{i=1}^{N} a_{_{elite}} \log(scores)`
| :green:`# optimization step`
| reset :math:`\text{optimizer}_\theta`
| :math:`\nabla_{\theta} L_{\pi_\theta}`
| step :math:`\text{optimizer}_\theta`
| :green:`# update learning rate`
| **IF** there is a :guilabel:`learning_rate_scheduler` **THEN**
|     step :math:`\text{scheduler}_\theta (\text{optimizer}_\theta)`

Basic usage
^^^^^^^^^^^

.. tabs::

    .. tab:: Standard implementation

        .. literalinclude:: ../snippets/agents_basic_usage.py
            :language: python
            :emphasize-lines: 2
            :start-after: [start-cem]
            :end-before: [end-cem]

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:data:: skrl.agents.torch.cem.cem.CEM_DEFAULT_CONFIG

.. literalinclude:: ../../../skrl/agents/torch/cem/cem.py
    :language: python
    :lines: 15-44
    :linenos:

Spaces and models
^^^^^^^^^^^^^^^^^

The implementation supports the following `Gym spaces <https://www.gymlibrary.dev/api/spaces>`_ / `Gymnasium spaces <https://gymnasium.farama.org/api/spaces>`_

.. list-table::
    :header-rows: 1

    * - Gym/Gymnasium spaces
      - .. centered:: Observation
      - .. centered:: Action
    * - Discrete
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\blacksquare`
    * - Box
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - Dict
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

The implementation uses 1 discrete function approximator. This function approximator (model) must be collected in a dictionary and passed to the constructor of the class under the argument :literal:`models`

.. list-table::
    :header-rows: 1

    * - Notation
      - Concept
      - Key
      - Input shape
      - Output shape
      - Type
    * - :math:`\pi(s)`
      - Policy
      - :literal:`"policy"`
      - observation
      - action
      - :ref:`Categorical <models_categorical>`

Support for advanced features is described in the next table

.. list-table::
    :header-rows: 1

    * - Feature
      - Support and remarks
    * - RNN support
      - \-

API
^^^

.. autoclass:: skrl.agents.torch.cem.cem.CEM
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__
