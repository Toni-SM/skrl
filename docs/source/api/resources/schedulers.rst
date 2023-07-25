Learning rate schedulers
========================

.. toctree::
    :hidden:

    KL Adaptive <schedulers/kl_adaptive>

Learning rate schedulers are techniques that adjust the learning rate over time to improve the performance of the agent.

.. raw:: html

    <br><hr>

.. list-table::
    :header-rows: 1

    * - Learning rate schedulers
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - :doc:`KL Adaptive <schedulers/kl_adaptive>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

|

**Implementation according to the ML framework:**

- **PyTorch**: The implemented schedulers inherit from the PyTorch :literal:`_LRScheduler` class. Visit `How to adjust learning rate <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_ in the PyTorch documentation for more details.

- **JAX**: The implemented schedulers must parameterize and return a function that maps step counts to values. Visit `Schedules <https://optax.readthedocs.io/en/latest/api.html#schedules>`_ in the Optax documentation for more details.

.. raw:: html

    <br>

Usage
-----

The learning rate scheduler usage is defined in each agent's configuration dictionary. The scheduler class is set under the :literal:`"learning_rate_scheduler"` key and its arguments are set under the :literal:`"learning_rate_scheduler_kwargs"` key as a keyword argument dictionary, without specifying the optimizer (first argument).

The following examples show how to set the scheduler for an agent:

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. tabs::

            .. tab:: PyTorch scheduler

                .. code-block:: python
                    :emphasize-lines: 2, 5-6

                    # import the scheduler class
                    from torch.optim.lr_scheduler import StepLR

                    cfg = DEFAULT_CONFIG.copy()
                    cfg["learning_rate_scheduler"] = StepLR
                    cfg["learning_rate_scheduler_kwargs"] = {"step_size": 1, "gamma": 0.9}

            .. tab:: skrl scheduler

                .. code-block:: python
                    :emphasize-lines: 2, 5-6

                    # import the scheduler class
                    from skrl.resources.schedulers.torch import KLAdaptiveLR

                    cfg = DEFAULT_CONFIG.copy()
                    cfg["learning_rate_scheduler"] = KLAdaptiveLR
                    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}

    .. group-tab:: |_4| |jax| |_4|

        .. tabs::

            .. tab:: JAX (Optax) scheduler

                .. code-block:: python
                    :emphasize-lines: 2, 5-6

                    # import the scheduler function
                    from optax import constant_schedule

                    cfg = DEFAULT_CONFIG.copy()
                    cfg["learning_rate_scheduler"] = constant_schedule
                    cfg["learning_rate_scheduler_kwargs"] = {"value": 1e-4}

            .. tab:: skrl scheduler

                .. code-block:: python
                    :emphasize-lines: 2, 5-6

                    # import the scheduler class
                    from skrl.resources.schedulers.jax import KLAdaptiveLR  # or kl_adaptive (Optax style)

                    cfg = DEFAULT_CONFIG.copy()
                    cfg["learning_rate_scheduler"] = KLAdaptiveLR
                    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
