Omniverse Isaac Gym environments
================================

.. image:: ../../_static/imgs/example_omniverse_isaacgym.png
    :width: 100%
    :align: center
    :alt: Omniverse Isaac Gym environments

.. raw:: html

    <br><br><hr>

Environments
------------

The repository https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs provides the example reinforcement learning environments for Omniverse Isaac Gym.

These environments can be easily loaded and configured by calling a single function provided with this library. This function also makes it possible to configure the environment from the command line arguments (see OmniIsaacGymEnvs's `configuration-and-command-line-arguments <https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs#configuration-and-command-line-arguments>`_) or from its parameters (:literal:`task_name`, :literal:`num_envs`, :literal:`headless`, and :literal:`cli_args`).

Additionally, multi-threaded environments can be loaded. These are designed to isolate the RL policy in a new thread, separate from the main simulation and rendering thread. Read more about it in the OmniIsaacGymEnvs framework documentation: `Multi-Threaded Environment Wrapper <https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/blob/220d34c6b68d3f7518c4aa008ae009d13cc60c03/docs/framework.md#multi-threaded-environment-wrapper>`_.

.. note::

    The command line arguments has priority over the function parameters.

.. note::

    Only the configuration related to the environment will be used. The configuration related to RL algorithms are discarded since they do not belong to this library.

.. note::

    Omniverse Isaac Gym environments implement a functionality to get their configuration from the command line. Setting the :literal:`headless` option from the trainer configuration will not work. In this case, it is necessary to set the load function's :literal:`headless` argument to True or to invoke the scripts as follows: :literal:`python script.py headless=True`.

.. raw:: html

    <br>

Usage
^^^^^

.. raw:: html

    <br>

Common environments
"""""""""""""""""""

In this approach, the RL algorithm maintains the main execution loop.

.. tabs::

    .. group-tab:: Function parameters

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-omniverse-isaac-gym-envs-parameters-torch]
                    :end-before: [end-omniverse-isaac-gym-envs-parameters-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-omniverse-isaac-gym-envs-parameters-jax]
                    :end-before: [end-omniverse-isaac-gym-envs-parameters-jax]

    .. group-tab:: Command line arguments (priority)

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-omniverse-isaac-gym-envs-cli-torch]
                    :end-before: [end-omniverse-isaac-gym-envs-cli-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-omniverse-isaac-gym-envs-cli-jax]
                    :end-before: [end-omniverse-isaac-gym-envs-cli-jax]

        Run the main script passing the configuration as command line arguments. For example:

        .. code-block::

            python main.py task=Cartpole

.. raw:: html

    <br>

Multi-threaded environments
"""""""""""""""""""""""""""

In this approach, the RL algorithm is executed on a secondary thread while the simulation and rendering is executed on the main thread.

.. tabs::

    .. group-tab:: Function parameters

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 1, 4, 7, 12, 15
                    :start-after: [start-omniverse-isaac-gym-envs-multi-threaded-parameters-torch]
                    :end-before: [end-omniverse-isaac-gym-envs-multi-threaded-parameters-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 1, 4, 7, 12, 15
                    :start-after: [start-omniverse-isaac-gym-envs-multi-threaded-parameters-jax]
                    :end-before: [end-omniverse-isaac-gym-envs-multi-threaded-parameters-jax]

    .. group-tab:: Command line arguments (priority)

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 1, 4, 7, 12, 15
                    :start-after: [start-omniverse-isaac-gym-envs-multi-threaded-cli-torch]
                    :end-before: [end-omniverse-isaac-gym-envs-multi-threaded-cli-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 1, 4, 7, 12, 15
                    :start-after: [start-omniverse-isaac-gym-envs-multi-threaded-cli-jax]
                    :end-before: [end-omniverse-isaac-gym-envs-multi-threaded-cli-jax]

        Run the main script passing the configuration as command line arguments. For example:

        .. code-block::

            python main.py task=Cartpole

.. raw:: html

    <br>

API
^^^

.. autofunction:: skrl.envs.loaders.torch.load_omniverse_isaacgym_env
