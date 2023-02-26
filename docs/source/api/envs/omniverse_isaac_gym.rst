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

The repository https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs provides the example reinforcement learning environments for Omniverse Isaac Gym

These environments can be easily loaded and configured by calling a single function provided with this library. This function also makes it possible to configure the environment from the command line arguments (see `configuration-and-command-line-arguments <https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs#configuration-and-command-line-arguments>`_) or from its parameters as a python dictionary

Additionally, multi-threaded environments can be loaded. These are designed to isolate the RL policy in a new thread, separate from the main simulation and rendering thread. Read more about it in the NVIDIA Omniverse Isaac Sim documentation: `Multi-Threaded Environment Wrapper <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_isaac_gym.html#multi-threaded-environment-wrapper>`_

.. note::

    The command line arguments has priority over the function parameters

.. note::

    Only the configuration related to the environment will be used. The configuration related to RL algorithms are discarded since they do not belong to this library

.. note::

    Omniverse Isaac Gym environments implement a functionality to get their configuration from the command line. Setting the :literal:`headless` option from the trainer configuration will not work. In this case, it is necessary to invoke the scripts as follows: :literal:`python script.py headless=True`

.. raw:: html

    <br>

Usage
^^^^^

.. raw:: html

    <br>

Common environments
"""""""""""""""""""

In this approach, the RL algorithm maintains the main execution loop

.. tabs::

    .. tab:: Function parameters

        .. code-block:: python
            :linenos:

            # import the environment loader
            from skrl.envs.torch import load_omniverse_isaacgym_env

            # load environment
            env = load_omniverse_isaacgym_env(task_name="Cartpole")

    .. tab:: Command line arguments (priority)

        .. code-block:: python
            :linenos:

            # import the environment loader
            from skrl.envs.torch import load_omniverse_isaacgym_env

            # load environment
            env = load_omniverse_isaacgym_env()

        Run the main script passing the configuration as command line arguments. For example:

        .. code-block::

            python main.py task=Cartpole

.. raw:: html

    <br>

Multi-threaded environments
"""""""""""""""""""""""""""

In this approach, the RL algorithm is executed on a secondary thread while the simulation and rendering is executed on the main thread

.. tabs::

    .. tab:: Function parameters

        .. code-block:: python
            :linenos:

            import threading

            # import the environment loader
            from skrl.envs.torch import load_omniverse_isaacgym_env

            # load environment
            env = load_omniverse_isaacgym_env(task_name="Cartpole", multi_threaded=True, timeout=30)

            ...

            # start training in a separate thread
            threading.Thread(target=trainer.train).start()

            # run the simulation in the main thread
            env.run()

    .. tab:: Command line arguments (priority)

        .. code-block:: python
            :linenos:

            import threading

            # import the environment loader
            from skrl.envs.torch import load_omniverse_isaacgym_env

            # load environment
            env = load_omniverse_isaacgym_env(multi_threaded=True, timeout=30)

            ...

            # start training in a separate thread
            threading.Thread(target=trainer.train).start()

            # run the simulation in the main thread
            env.run()

        Run the main script passing the configuration as command line arguments. For example:

        .. code-block::

            python main.py task=Cartpole

.. raw:: html

    <br>

API
^^^

.. autofunction:: skrl.envs.torch.loaders.load_omniverse_isaacgym_env
