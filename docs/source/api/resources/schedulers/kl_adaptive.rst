KL Adaptive
===========

Adjust the learning rate according to the value of the Kullback-Leibler (KL) divergence.

.. raw:: html

    <br><hr>

Algorithm
---------

.. raw:: html

    <br>

Algorithm implementation
^^^^^^^^^^^^^^^^^^^^^^^^

The learning rate (:math:`\eta`) at each step is modified as follows:

| **IF** :math:`\; KL >` :guilabel:`kl_factor` :guilabel:`kl_threshold` **THEN**
|     :math:`\eta_{t + 1} = \max(\eta_t \,/` :guilabel:`lr_factor` :math:`,` :guilabel:`min_lr` :math:`)`
| **IF** :math:`\; KL <` :guilabel:`kl_threshold` :math:`/` :guilabel:`kl_factor` **THEN**
|     :math:`\eta_{t + 1} = \min(` :guilabel:`lr_factor` :math:`\eta_t,` :guilabel:`max_lr` :math:`)`

.. raw:: html

    <br>

Usage
-----

The learning rate scheduler usage is defined in each agent's configuration dictionary. The scheduler class is set under the :literal:`"learning_rate_scheduler"` key and its arguments are set under the :literal:`"learning_rate_scheduler_kwargs"` key as a keyword argument dictionary, without specifying the optimizer (first argument). The following examples show how to set the scheduler for an agent:

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. code-block:: python
            :emphasize-lines: 2, 5-6

            # import the scheduler class
            from skrl.resources.schedulers.torch import KLAdaptiveLR

            cfg = DEFAULT_CONFIG.copy()
            cfg["learning_rate_scheduler"] = KLAdaptiveLR
            cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}

    .. group-tab:: |_4| |jax| |_4|

        .. code-block:: python
            :emphasize-lines: 2, 5-6

            # import the scheduler class
            from skrl.resources.schedulers.jax import KLAdaptiveLR  # or kl_adaptive (Optax style)

            cfg = DEFAULT_CONFIG.copy()
            cfg["learning_rate_scheduler"] = KLAdaptiveLR
            cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.resources.schedulers.torch.kl_adaptive.KLAdaptiveLR
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.resources.schedulers.jax.kl_adaptive.KLAdaptiveLR
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__
