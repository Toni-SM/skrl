Isaac Gym environments
======================

.. image:: ../../_static/imgs/example_isaacgym.png
    :width: 100%
    :align: center
    :alt: Omniverse Isaac Gym environments

.. raw:: html

    <br><br><hr>

Environments (preview 4)
------------------------

The repository https://github.com/NVIDIA-Omniverse/IsaacGymEnvs provides the example reinforcement learning environments for Isaac Gym (preview 4).

With the release of Isaac Gym (preview 4), NVIDIA developers provide an easy-to-use API for creating/loading preset vectorized environments (see IsaacGymEnvs's  `creating-an-environment <https://github.com/NVIDIA-Omniverse/IsaacGymEnvs#creating-an-environment>`_).

.. tabs::

    .. tab:: Easy-to-use API from NVIDIA

        .. literalinclude:: ../../snippets/loaders.py
            :language: python
            :start-after: [start-isaac-gym-envs-preview-4-api]
            :end-before: [end-isaac-gym-envs-preview-4-api]

Nevertheless, in order to maintain the loading style of previous versions, **skrl** provides its own implementation for loading such environments. The environments can be easily loaded and configured by calling a single function provided with this library. This function also makes it possible to configure the environment from the command line arguments (see IsaacGymEnvs's `configuration-and-command-line-arguments <https://github.com/NVIDIA-Omniverse/IsaacGymEnvs#configuration-and-command-line-arguments>`_) or from its parameters (:literal:`task_name`, :literal:`num_envs`, :literal:`headless`, and :literal:`cli_args`).

.. note::

    Only the configuration related to the environment will be used. The configuration related to RL algorithms are discarded since they do not belong to this library.

.. note::

    Isaac Gym environments implement a functionality to get their configuration from the command line. Setting the :literal:`headless` option from the trainer configuration will not work. In this case, it is necessary to set the load function's :literal:`headless` argument to True or to invoke the scripts as follows: :literal:`python script.py headless=True`.

.. raw:: html

    <br>

Usage
^^^^^

.. tabs::

    .. group-tab:: Function parameters

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-isaac-gym-envs-preview-4-parameters-torch]
                    :end-before: [end-isaac-gym-envs-preview-4-parameters-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-isaac-gym-envs-preview-4-parameters-jax]
                    :end-before: [end-isaac-gym-envs-preview-4-parameters-jax]

    .. group-tab:: Command line arguments (priority)

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-isaac-gym-envs-preview-4-cli-torch]
                    :end-before: [end-isaac-gym-envs-preview-4-cli-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-isaac-gym-envs-preview-4-cli-jax]
                    :end-before: [end-isaac-gym-envs-preview-4-cli-jax]

        Run the main script passing the configuration as command line arguments. For example:

        .. code-block::

            python main.py task=Cartpole

.. raw:: html

    <br>

API
^^^

.. autofunction:: skrl.envs.loaders.torch.load_isaacgym_env_preview4

.. raw:: html

    <br><hr>

Environments (preview 3)
------------------------

The repository https://github.com/NVIDIA-Omniverse/IsaacGymEnvs provides the example reinforcement learning environments for Isaac Gym (preview 3).

These environments can be easily loaded and configured by calling a single function provided with this library. This function also makes it possible to configure the environment from the command line arguments (see IsaacGymEnvs's `configuration-and-command-line-arguments <https://github.com/NVIDIA-Omniverse/IsaacGymEnvs#configuration-and-command-line-arguments>`_) or from its parameters (:literal:`task_name`, :literal:`num_envs`, :literal:`headless`, and :literal:`cli_args`).

.. note::

    Only the configuration related to the environment will be used. The configuration related to RL algorithms are discarded since they do not belong to this library.

.. note::

    Isaac Gym environments implement a functionality to get their configuration from the command line. Setting the :literal:`headless` option from the trainer configuration will not work. In this case, it is necessary to set the load function's :literal:`headless` argument to True or to invoke the scripts as follows: :literal:`python script.py headless=True`.

.. raw:: html

    <br>

Usage
^^^^^

.. tabs::

    .. group-tab:: Function parameters

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-isaac-gym-envs-preview-3-parameters-torch]
                    :end-before: [end-isaac-gym-envs-preview-3-parameters-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-isaac-gym-envs-preview-3-parameters-jax]
                    :end-before: [end-isaac-gym-envs-preview-3-parameters-jax]

    .. group-tab:: Command line arguments (priority)

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-isaac-gym-envs-preview-3-cli-torch]
                    :end-before: [end-isaac-gym-envs-preview-3-cli-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-isaac-gym-envs-preview-3-cli-jax]
                    :end-before: [end-isaac-gym-envs-preview-3-cli-jax]

        Run the main script passing the configuration as command line arguments. For example:

        .. code-block::

            python main.py task=Cartpole

.. raw:: html

    <br>

API
^^^

.. autofunction:: skrl.envs.loaders.torch.load_isaacgym_env_preview3

.. raw:: html

    <br><hr>

Environments (preview 2)
------------------------

The example reinforcement learning environments for Isaac Gym (preview 2) are located within the same package (in the :code:`python/rlgpu` directory).

These environments can be easily loaded and configured by calling a single function provided with this library. This function also makes it possible to configure the environment from the command line arguments or from its parameters (:literal:`task_name`, :literal:`num_envs`, :literal:`headless`, and :literal:`cli_args`).

.. note::

    Isaac Gym environments implement a functionality to get their configuration from the command line. Setting the :literal:`headless` option from the trainer configuration will not work. In this case, it is necessary to set the load function's :literal:`headless` argument to True or to invoke the scripts as follows: :literal:`python script.py --headless`.

.. raw:: html

    <br>

Usage
^^^^^

.. tabs::

    .. group-tab:: Function parameters

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-isaac-gym-envs-preview-2-parameters-torch]
                    :end-before: [end-isaac-gym-envs-preview-2-parameters-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-isaac-gym-envs-preview-2-parameters-jax]
                    :end-before: [end-isaac-gym-envs-preview-2-parameters-jax]

    .. group-tab:: Command line arguments (priority)

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-isaac-gym-envs-preview-2-cli-torch]
                    :end-before: [end-isaac-gym-envs-preview-2-cli-torch]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/loaders.py
                    :language: python
                    :emphasize-lines: 2, 5
                    :start-after: [start-isaac-gym-envs-preview-2-cli-jax]
                    :end-before: [end-isaac-gym-envs-preview-2-cli-jax]

        Run the main script passing the configuration as command line arguments. For example:

        .. code-block::

            python main.py --task Cartpole

.. raw:: html

    <br>

API
^^^

.. autofunction:: skrl.envs.loaders.torch.load_isaacgym_env_preview2
