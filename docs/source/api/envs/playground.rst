Playground environments
=======================

|br| |hr|

Overview
--------

The repository https://github.com/google-deepmind/mujoco_playground provides the MuJoCo Playground environments.

These environments can be easily loaded and configured by calling a single function provided with this library.
Such function also makes it possible to configure the environment from the command line arguments or from its parameters.

.. note::

    The command line arguments has priority over the function parameters.

    Run the following command to list all the available MuJoCo Playground environments:

    .. code-block:: bash

        python -c "import mujoco_playground; print(mujoco_playground.registry.ALL_ENVS)"

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

            python script.py --task CartpoleBalance --num_envs 1024 --episode_length 300

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
