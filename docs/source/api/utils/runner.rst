Runner
======

Utility that configures and instantiates skrl's components to run training/evaluation workflows in a few lines of code.

.. raw:: html

    <br><hr>

Usage
-----

.. hint::

    The ``Runner`` classes encapsulates, and greatly simplifies, the definitions and instantiations needed to execute RL tasks.
    However, such simplification hides and makes difficult the modification and readability of the code (models, agents, etc.).

    For more control and readability over the RL system setup refer to the :doc:`Examples <../../intro/examples>` section's training scripts (**recommended!**)

.. raw:: html

    <br>

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. tabs::

            .. group-tab:: Python code

                .. literalinclude:: ../../snippets/runner.py
                    :language: python
                    :emphasize-lines: 1, 10, 13, 16
                    :start-after: [start-runner-train-torch]
                    :end-before: [end-runner-train-torch]

            .. group-tab:: Example .yaml file (PPO)

                .. literalinclude:: ../../snippets/runner.py
                    :language: yaml
                    :start-after: [start-cfg-yaml]
                    :end-before: [end-cfg-yaml]

    .. group-tab:: |_4| |jax| |_4|

        .. tabs::

            .. group-tab:: Python code

                .. literalinclude:: ../../snippets/runner.py
                    :language: python
                    :emphasize-lines: 1, 10, 13, 16
                    :start-after: [start-runner-train-jax]
                    :end-before: [end-runner-train-jax]

            .. group-tab:: Example .yaml file (PPO)

                .. literalinclude:: ../../snippets/runner.py
                    :language: yaml
                    :start-after: [start-cfg-yaml]
                    :end-before: [end-cfg-yaml]

API (PyTorch)
-------------

.. autoclass:: skrl.utils.runner.torch.Runner
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.utils.runner.jax.Runner
    :show-inheritance:
    :members:

    .. automethod:: __init__
