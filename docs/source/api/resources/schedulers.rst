Learning rate schedulers
========================

.. toctree::
    :hidden:

    KL Adaptive <schedulers/kl_adaptive>

Learning rate schedulers are techniques that adjust the learning rate over time to improve the performance of the agent.

.. raw:: html

    <br><hr>

The implemented schedulers inherit from the PyTorch :literal:`_LRScheduler` class. Visit `how to adjust learning rate <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_ in the PyTorch documentation for more details

.. raw:: html

    <br>

Usage
-----

The learning rate scheduler usage is defined in each agent's configuration dictionary. The scheduler class is set under the :literal:`"learning_rate_scheduler"` key and its arguments are set under the :literal:`"learning_rate_scheduler_kwargs"` key as a keyword argument dictionary, without specifying the optimizer (first argument). The following examples show how to set the scheduler for an agent:

.. tabs::

    .. tab:: PyTorch scheduler

        .. code-block:: python
            :emphasize-lines: 5-6

            # import the scheduler class
            from torch.optim.lr_scheduler import StepLR

            cfg = DEFAULT_CONFIG.copy()
            cfg["learning_rate_scheduler"] = StepLR
            cfg["learning_rate_scheduler_kwargs"] = {"step_size": 1, "gamma": 0.9}

    .. tab:: skrl scheduler

        .. code-block:: python
            :emphasize-lines: 5-6

            # import the scheduler class
            from skrl.resources.schedulers.torch import KLAdaptiveRL

            cfg = DEFAULT_CONFIG.copy()
            cfg["learning_rate_scheduler"] = KLAdaptiveRL
            cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
