Cross-Entropy Method (CEM)
==========================

.. raw:: html

    <br><hr>

Algorithm
---------

.. raw:: html

    <br>

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

| Main notation/symbols:
|   - policy function approximator (:math:`\pi_\theta`)
|   - states (:math:`s`), actions (:math:`a`), rewards (:math:`r`), next states (:math:`s'`), dones (:math:`d`)
|   - loss (:math:`L`)

.. raw:: html

    <br>

Decision making
"""""""""""""""

|
| :literal:`act(...)`
| :math:`a \leftarrow \pi_\theta(s)`

.. raw:: html

    <br>

Learning algorithm
""""""""""""""""""

|
| :literal:`_update(...)`
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

.. raw:: html

    <br>

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

.. raw:: html

    <br>

Configuration and hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../../../../skrl/agents/torch/cem/cem.py
    :language: python
    :start-after: [start-config-dict-torch]
    :end-before: [end-config-dict-torch]

.. raw:: html

    <br>

Spaces
^^^^^^

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

.. raw:: html

    <br>

Models
^^^^^^

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

.. raw:: html

    <br>

Features
^^^^^^^^

Support for advanced features is described in the next table

.. list-table::
    :header-rows: 1

    * - Feature
      - Support and remarks
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - RNN support
      - \-
      - .. centered:: :math:`\square`
      - .. centered:: :math:`\square`

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.agents.torch.cem.CEM_DEFAULT_CONFIG

.. autoclass:: skrl.agents.torch.cem.CEM
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.agents.jax.cem.CEM_DEFAULT_CONFIG

.. autoclass:: skrl.agents.jax.cem.CEM
    :undoc-members:
    :show-inheritance:
    :private-members: _update
    :members:

    .. automethod:: __init__
