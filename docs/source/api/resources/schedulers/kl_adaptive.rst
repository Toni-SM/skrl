:tocdepth: 4

KL Adaptive
===========

Adjust the learning rate according to the value of the Kullback-Leibler (KL) divergence.

|br| |hr|

Algorithm
---------

|

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

The learning rate (:math:`\eta`) at each step is modified as follows:

| **IF** :math:`\; KL >` :guilabel:`kl_factor` :guilabel:`kl_threshold` **THEN**
|     :math:`\eta_{t + 1} = \max(\eta_t \,/` :guilabel:`lr_factor` :math:`,` :guilabel:`min_lr` :math:`)`
| **IF** :math:`\; KL <` :guilabel:`kl_threshold` :math:`/` :guilabel:`kl_factor` **THEN**
|     :math:`\eta_{t + 1} = \min(` :guilabel:`lr_factor` :math:`\eta_t,` :guilabel:`max_lr` :math:`)`

|

Usage
-----

The learning rate scheduler usage is defined in each agent's configuration.
The scheduler class is set under the :literal:`learning_rate_scheduler` key and its arguments are set under
the :literal:`learning_rate_scheduler_kwargs` key, as a Python dictionary, without specifying the optimizer instance.

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. literalinclude:: ../../../snippets/schedulers.py
            :language: python
            :emphasize-lines: 2, 6-7
            :start-after: [start-scheduler-kl-adaptive-torch]
            :end-before: [end-scheduler-kl-adaptive-torch]

    .. group-tab:: |_4| |jax| |_4|

        .. literalinclude:: ../../../snippets/schedulers.py
            :language: python
            :emphasize-lines: 2, 6-7
            :start-after: [start-scheduler-kl-adaptive-jax]
            :end-before: [end-scheduler-kl-adaptive-jax]

    .. group-tab:: |_4| |warp| |_4|

        .. literalinclude:: ../../../snippets/schedulers.py
            :language: python
            :emphasize-lines: 2, 6-7
            :start-after: [start-scheduler-kl-adaptive-warp]
            :end-before: [end-scheduler-kl-adaptive-warp]

|

API
---

|

PyTorch
^^^^^^^

.. automodule:: skrl.resources.schedulers.torch.kl_adaptive
.. autosummary::
    :nosignatures:

    KLAdaptiveLR

.. autoclass:: skrl.resources.schedulers.torch.kl_adaptive.KLAdaptiveLR
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

JAX
^^^

.. automodule:: skrl.resources.schedulers.jax.kl_adaptive
.. autosummary::
    :nosignatures:

    KLAdaptiveLR

.. autofunction:: skrl.resources.schedulers.jax.kl_adaptive.KLAdaptiveLR

|

Warp
^^^^

.. automodule:: skrl.resources.schedulers.warp.kl_adaptive
.. autosummary::
    :nosignatures:

    KLAdaptiveLR

.. autofunction:: skrl.resources.schedulers.warp.kl_adaptive.KLAdaptiveLR
