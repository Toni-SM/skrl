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
|     :math:`\eta_{t + 1} = \max(` :guilabel:`lr_factor` :math:`^{-1} \; \eta_t,` :guilabel:`min_lr` :math:`)`
| **IF** :math:`\; KL <` :guilabel:`kl_factor` :math:`^{-1}` :guilabel:`kl_threshold` **THEN**
|     :math:`\eta_{t + 1} = \min(` :guilabel:`lr_factor` :math:`\eta_t,` :guilabel:`max_lr` :math:`)`

.. raw:: html

    <br>

Usage
-----

The learning rate scheduler usage is defined in each agent's configuration dictionary. The scheduler class is set under the :literal:`"learning_rate_scheduler"` key and its arguments are set under the :literal:`"learning_rate_scheduler_kwargs"` key as a keyword argument dictionary, without specifying the optimizer (first argument). The following examples show how to set the scheduler for an agent:

.. tabs::

    .. tab:: Scheduler

        .. code-block:: python
            :emphasize-lines: 5-6

            # import the scheduler class
            from skrl.resources.schedulers.torch import KLAdaptiveRL

            cfg = DEFAULT_CONFIG.copy()
            cfg["learning_rate_scheduler"] = KLAdaptiveRL
            cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}

.. raw:: html

    <br>

API
---

.. autoclass:: skrl.resources.schedulers.torch.kl_adaptive.KLAdaptiveRL
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__
