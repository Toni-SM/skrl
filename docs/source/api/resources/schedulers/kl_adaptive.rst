KL Adaptive
===========

Algorithm implementation
------------------------

The learning rate (:math:`\eta`) at each step is modified as follows:

| **IF** :math:`\; KL >` :guilabel:`kl_factor` :guilabel:`kl_threshold` **THEN**
|     :math:`\eta_{t + 1} = \max(` :guilabel:`lr_factor` :math:`^{-1} \; \eta_t,` :guilabel:`min_lr` :math:`)`
| **IF** :math:`\; KL <` :guilabel:`kl_factor` :math:`^{-1}` :guilabel:`kl_threshold` **THEN**
|     :math:`\eta_{t + 1} = \min(` :guilabel:`lr_factor` :math:`\eta_t,` :guilabel:`max_lr` :math:`)`

API
---

.. autoclass:: skrl.resources.schedulers.torch.kl_adaptive.KLAdaptiveRL
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__
