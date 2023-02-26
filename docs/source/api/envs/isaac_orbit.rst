Isaac Orbit environments
========================

.. image:: ../../_static/imgs/example_isaac_orbit.png
    :width: 100%
    :align: center
    :alt: Isaac Orbit environments

.. raw:: html

    <br><br><hr>

Environments
------------

The repository https://github.com/NVIDIA-Omniverse/Orbit provides the example reinforcement learning environments for Isaac orbit

These environments can be easily loaded and configured by calling a single function provided with this library. This function also makes it possible to configure the environment from the command line arguments (see `Running an RL environment <https://isaac-orbit.github.io/orbit/source/tutorials/04_gym_env.html>`_) or from its parameters as a python dictionary

.. note::

    The command line arguments has priority over the function parameters

.. note::

    Isaac Orbit environments implement a functionality to get their configuration from the command line. Setting the :literal:`headless` option from the trainer configuration will not work. In this case, it is necessary to invoke the scripts as follows: :literal:`orbit -p script.py --headless`

.. raw:: html

    <br>

Usage
^^^^^

.. tabs::

    .. tab:: Function parameters

        .. code-block:: python
            :linenos:

            # import the environment loader
            from skrl.envs.torch import load_isaac_orbit_env

            # load environment
            env = load_isaac_orbit_env(task_name="Isaac-Cartpole-v0")

    .. tab:: Command line arguments (priority)

        .. code-block:: python
            :linenos:

            # import the environment loader
            from skrl.envs.torch import load_isaac_orbit_env

            # load environment
            env = load_isaac_orbit_env()

        Run the main script passing the configuration as command line arguments. For example:

        .. code-block::

            orbit -p main.py --task Isaac-Cartpole-v0

.. raw:: html

    <br>

API
^^^

.. autofunction:: skrl.envs.torch.loaders.load_isaac_orbit_env
