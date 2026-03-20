:tocdepth: 4

Cross-Entropy Method (CEM)
==========================

|br| |hr|

Algorithm
---------

|

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - policy function approximator (:math:`\pi_\theta`)
|   - states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next states (:math:`s'`), terminated (:math:`d_{_{end}}`), truncated (:math:`d_{_{timeout}}`)
|   - loss (:math:`L`)

|

Decision making
"""""""""""""""

|
| :literal:`act(...)`
| :math:`a \leftarrow \pi_\theta(s)`

|

Learning algorithm
""""""""""""""""""

|
| :literal:`_update(...)`
| :green:`# sample all memory`
| :math:`s, a, r \leftarrow` states, actions, rewards
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

|

Usage
-----

.. tabs::

    .. tab:: Standard implementation

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [torch-start-cem]
                    :end-before: [torch-end-cem]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/agents_basic_usage.py
                    :language: python
                    :emphasize-lines: 2
                    :start-after: [jax-start-cem]
                    :end-before: [jax-end-cem]

|

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1

    * - Dataclass
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - ``CEM_CFG``
      - :py:class:`~skrl.agents.torch.cem.CEM_CFG`
      - :py:class:`~skrl.agents.jax.cem.CEM_CFG`
      -

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
      - .. centered:: :math:`\blacksquare`
    * - Box
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`
    * - Dict
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\square`

|

Models
^^^^^^

The implementation uses 1 discrete function approximator.
This function approximator (model) must be collected in a dictionary and passed to the constructor of the class
under the argument :literal:`models`.

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
      - :ref:`Categorical <models_categorical>` /
        |br| :ref:`Multi-Categorical <models_multicategorical>`

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
      - \-
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`

|

API
---

|

PyTorch
^^^^^^^

.. automodule:: skrl.agents.torch.cem
.. autosummary::
    :nosignatures:

    CEM_CFG
    CEM

.. autoclass:: skrl.agents.torch.cem.CEM_CFG
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.agents.torch.cem.CEM
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

JAX
^^^

.. automodule:: skrl.agents.jax.cem
.. autosummary::
    :nosignatures:

    CEM_CFG
    CEM

.. autoclass:: skrl.agents.jax.cem.CEM_CFG
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

.. autoclass:: skrl.agents.jax.cem.CEM
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
