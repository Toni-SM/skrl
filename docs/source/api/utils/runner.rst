:tocdepth: 4

Runner
======

Utility that configures and instantiates skrl's components to run training/evaluation workflows in a few lines of code.

|br| |hr|

Usage
-----

.. hint::

    The ``Runner`` classes encapsulates, and greatly simplifies, the definitions and instantiations
    needed to execute RL tasks. However, such simplification hides and makes difficult the modification
    and readability of the code (models, agents, etc.).

    For more control and readability over the RL system setup refer to the :doc:`Examples <../../intro/examples>`
    section's training scripts (**recommended!**).

|

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. tabs::

            .. group-tab:: Python code

                .. literalinclude:: ../../snippets/runner.txt
                    :language: python
                    :emphasize-lines: 1, 10, 13, 16
                    :start-after: [start-runner-train-torch]
                    :end-before: [end-runner-train-torch]

            .. group-tab:: Example .yaml file (PPO)

                .. literalinclude:: ../../snippets/runner.txt
                    :language: yaml
                    :start-after: [start-cfg-yaml]
                    :end-before: [end-cfg-yaml]

    .. group-tab:: |_4| |jax| |_4|

        .. tabs::

            .. group-tab:: Python code

                .. literalinclude:: ../../snippets/runner.txt
                    :language: python
                    :emphasize-lines: 1, 10, 13, 16
                    :start-after: [start-runner-train-jax]
                    :end-before: [end-runner-train-jax]

            .. group-tab:: Example .yaml file (PPO)

                .. literalinclude:: ../../snippets/runner.txt
                    :language: yaml
                    :start-after: [start-cfg-yaml]
                    :end-before: [end-cfg-yaml]

    .. group-tab:: |_4| |warp| |_4|

        .. tabs::

            .. group-tab:: Python code

                .. literalinclude:: ../../snippets/runner.txt
                    :language: python
                    :emphasize-lines: 1, 10, 13, 16
                    :start-after: [start-runner-train-warp]
                    :end-before: [end-runner-train-warp]

            .. group-tab:: Example .yaml file (PPO)

                .. literalinclude:: ../../snippets/runner.txt
                    :language: yaml
                    :start-after: [start-cfg-yaml]
                    :end-before: [end-cfg-yaml]

|

API
---

|

PyTorch
^^^^^^^

.. automodule:: skrl.utils.runner.torch
.. autosummary::
    :nosignatures:

    Runner

.. autoclass:: skrl.utils.runner.torch.Runner
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

JAX
^^^

.. automodule:: skrl.utils.runner.jax
.. autosummary::
    :nosignatures:

    Runner

.. autoclass:: skrl.utils.runner.jax.Runner
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

|

Warp
^^^^

.. automodule:: skrl.utils.runner.warp
.. autosummary::
    :nosignatures:

    Runner

.. autoclass:: skrl.utils.runner.warp.Runner
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
