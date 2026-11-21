Playground environments
=======================

|br| |hr|

Overview
--------

The repository https://github.com/google-deepmind/mujoco_playground provides the MuJoCo Playground environments.

These environments can be easily loaded and configured by calling a single function provided with this library.
Such function also makes it possible to configure the environment from the command line arguments or from its parameters.

.. important::

    MuJoCo Playground 1.0.0 or later is required (``pip install "playground>=1.0.0"``, Python 3.12 or later).
    Since version 1.0.0, MuJoCo Playground environments are implemented on top of PyTorch and MuJoCo Warp
    (the previous JAX/MJX-based environments are not supported). Therefore, PyTorch is required regardless
    of the ML framework used by skrl.

.. note::

    The command line arguments has priority over the function parameters.

    Run the following command to list all the available MuJoCo Playground environments:

    .. code-block:: bash

        python -c "from mujoco_playground.rl import env_registry; print(list(env_registry._REGISTRY))"

|

Usage
-----

The following snippets show how to load MuJoCo Playground environments:

|

.. tabs::

    .. tab:: Function parameters

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-playground-envs-parameters-torch]
                    :end-before: [end-playground-envs-parameters-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-playground-envs-parameters-jax]
                    :end-before: [end-playground-envs-parameters-jax]

            .. group-tab:: |_4| |warp| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-playground-envs-parameters-warp]
                    :end-before: [end-playground-envs-parameters-warp]

    .. tab:: Command line arguments (priority)

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-playground-envs-cli-torch]
                    :end-before: [end-playground-envs-cli-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-playground-envs-cli-jax]
                    :end-before: [end-playground-envs-cli-jax]

            .. group-tab:: |_4| |warp| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-playground-envs-cli-warp]
                    :end-before: [end-playground-envs-cli-warp]

        Run the main script passing the configuration as command line arguments. For example:

        .. code-block::

            python script.py --task CartpoleBalance --num_envs 1024 --device cuda:0

|

API
---

|

PyTorch
^^^^^^^

.. autofunction:: skrl.envs.loaders.torch.load_playground_env

|

JAX
^^^

.. autofunction:: skrl.envs.loaders.jax.load_playground_env

|

Warp
^^^^

.. autofunction:: skrl.envs.loaders.warp.load_playground_env
