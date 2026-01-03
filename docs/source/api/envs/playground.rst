Playground environments
=======================

.. image:: ../../_static/imgs/example_playground.png
    :width: 100%
    :align: center
    :alt: Playground environments

.. raw:: html

    <br><br><hr>

Environments
------------

The repository https://github.com/google-deepmind/mujoco_playground provides the MuJoCo Playground environments.

These environments can be easily loaded and configured by calling a single function provided with this library. This function also makes it possible to configure the environment from the command line arguments or from its parameters (:literal:`task_name`, :literal:`num_envs`, among others).

.. note::

    The command line arguments has priority over the function parameters.

.. raw:: html

    <br>

Usage
^^^^^

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

            python script.py --task CartpoleBalance

.. raw:: html

    <br>

API
^^^

.. autofunction:: skrl.envs.loaders.jax.load_playground_env
