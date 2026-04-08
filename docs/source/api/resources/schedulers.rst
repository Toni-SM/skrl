:tocdepth: 4

.. _learning_rate_schedulers:

Learning rate schedulers
========================

.. toctree::
    :hidden:

    KL Adaptive <schedulers/kl_adaptive>

Learning rate schedulers are techniques that adjust the learning rate over time to improve the performance of the agent.

|br| |hr|

Implemented schedulers
----------------------

The following table lists the implemented schedulers and their support for different frameworks.

.. list-table::
    :header-rows: 1

    * - Learning rate schedulers
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`KL Adaptive <schedulers/kl_adaptive>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

|

Implementation details according to the ML framework:

* **PyTorch**: The implemented schedulers inherit from the PyTorch :literal:`_LRScheduler` class.
  Visit `How to adjust learning rate <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_
  in the PyTorch documentation for more details.

* **JAX**: The implemented schedulers must parameterize and return a function that maps step counts to values.
  Visit `Optimizer Schedules <https://optax.readthedocs.io/en/latest/api/optimizer_schedules.html>`_
  in the Optax documentation for more details.

* **Warp**: The implemented schedulers must parameterize and return a function that maps step counts to values.

|

Usage
-----

The learning rate scheduler usage is defined in each agent's configuration.
The scheduler class is set under the :literal:`learning_rate_scheduler` key and its arguments are set under
the :literal:`learning_rate_scheduler_kwargs` key, as a Python dictionary, without specifying the optimizer instance.

The following examples show how to set the scheduler for an agent, using either a third-party scheduler
(from the ML framework) or a native scheduler (from *skrl*):

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. tabs::

            .. group-tab:: PyTorch scheduler

                .. literalinclude:: ../../snippets/schedulers.py
                    :language: python
                    :emphasize-lines: 2, 6-7
                    :start-after: [start-3rd-party-torch]
                    :end-before: [end-3rd-party-torch]

            .. group-tab:: skrl scheduler

                .. literalinclude:: ../../snippets/schedulers.py
                    :language: python
                    :emphasize-lines: 2, 6-7
                    :start-after: [start-native-torch]
                    :end-before: [end-native-torch]

    .. group-tab:: |_4| |jax| |_4|

        .. tabs::

            .. group-tab:: JAX (Optax) scheduler

                .. literalinclude:: ../../snippets/schedulers.py
                    :language: python
                    :emphasize-lines: 2, 6-7
                    :start-after: [start-3rd-party-jax]
                    :end-before: [end-3rd-party-jax]

            .. group-tab:: skrl scheduler

                .. literalinclude:: ../../snippets/schedulers.py
                    :language: python
                    :emphasize-lines: 2, 6-7
                    :start-after: [start-native-jax]
                    :end-before: [end-native-jax]

    .. group-tab:: |_4| |warp| |_4|

        .. tabs::

            .. group-tab:: skrl scheduler

                .. literalinclude:: ../../snippets/schedulers.py
                    :language: python
                    :emphasize-lines: 2, 6-7
                    :start-after: [start-native-warp]
                    :end-before: [end-native-warp]
