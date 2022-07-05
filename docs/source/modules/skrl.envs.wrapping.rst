Wrapping
========

This library works with a common API to interact with the following RL environments:

* `OpenAI Gym <https://gym.openai.com/>`_ 
* `DeepMind <https://github.com/deepmind/dm_env>`_
* `NVIDIA Isaac Gym <https://developer.nvidia.com/isaac-gym>`_ (preview 2, 3 and 4)
* `NVIDIA Omniverse Isaac Gym <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_isaac_gym.html>`_

To operate with them and to support interoperability between these non-compatible interfaces, a **wrapping mechanism is provided** as shown in the diagram below

.. image:: ../_static/imgs/wrapping.svg
      :width: 100%
      :align: center
      :alt: Environment wrapping

.. raw:: html

    <br><br>

Basic usage
^^^^^^^^^^^

.. tabs::

    .. tab:: Omniverse Isaac Gym

        .. tabs::

            .. tab:: Common environment

                .. code-block:: python
                    :linenos:

                    # import the environment wrapper and loader
                    from skrl.envs.torch import wrap_env
                    from skrl.envs.torch import load_omniverse_isaacgym_env

                    # load the environment
                    env = load_omniverse_isaacgym_env(task_name="Cartpole")

                    # wrap the environment
                    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="omniverse-isaacgym")'

            .. tab:: Multi-threaded environment

                .. code-block:: python
                    :linenos:

                    # import the environment wrapper and loader
                    from skrl.envs.torch import wrap_env
                    from skrl.envs.torch import load_omniverse_isaacgym_env

                    # load the multi-threaded environment
                    env = load_omniverse_isaacgym_env(task_name="Cartpole", multi_threaded=True, timeout=30)

                    # wrap the environment
                    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="omniverse-isaacgym")'

    .. tab:: Isaac Gym

        .. tabs::

            .. tab:: Preview 4 (isaacgymenvs.make)
            
                .. code-block:: python
                    :linenos:

                    import isaacgymenvs

                    # import the environment wrapper
                    from skrl.envs.torch import wrap_env

                    # create/load the environment using the easy-to-use API from NVIDIA
                    env = isaacgymenvs.make(seed=0, 
                                            task="Cartpole", 
                                            num_envs=512, 
                                            sim_device="cuda:0",
                                            rl_device="cuda:0",
                                            graphics_device_id=0,
                                            headless=False)

                    # wrap the environment
                    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview4")'

            .. tab:: Preview 4
            
                .. code-block:: python
                    :linenos:

                    # import the environment wrapper and loader
                    from skrl.envs.torch import wrap_env
                    from skrl.envs.torch import load_isaacgym_env_preview4

                    # load the environment
                    env = load_isaacgym_env_preview4(task_name="Cartpole")

                    # wrap the environment
                    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview4")'

            .. tab:: Preview 3
            
                .. code-block:: python
                    :linenos:

                    # import the environment wrapper and loader
                    from skrl.envs.torch import wrap_env
                    from skrl.envs.torch import load_isaacgym_env_preview3

                    # load the environment
                    env = load_isaacgym_env_preview3(task_name="Cartpole")

                    # wrap the environment
                    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview3")'

            .. tab:: Preview 2
            
                .. code-block:: python
                    :linenos:

                    # import the environment wrapper and loader
                    from skrl.envs.torch import wrap_env
                    from skrl.envs.torch import load_isaacgym_env_preview2

                    # load the environment
                    env = load_isaacgym_env_preview2(task_name="Cartpole")

                    # wrap the environment
                    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview2")'

    .. tab:: OpenAI Gym
   
        .. code-block:: python
            :linenos:

            # import the environment wrapper and gym
            from skrl.envs.torch import wrap_env
            import gym

            # load environment
            env = gym.make('Pendulum-v1')

            # wrap the environment
            env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'

    .. tab:: DeepMind
   
        .. code-block:: python
            :linenos:

            # import the environment wrapper and the deepmind suite
            from skrl.envs.torch import wrap_env
            from dm_control import suite

            # load environment
            env = suite.load(domain_name="cartpole", task_name="swingup")

            # wrap the environment
            env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="dm")'

.. raw:: html

    <hr>

API
^^^

.. autofunction:: skrl.envs.torch.wrappers.wrap_env

.. raw:: html

    <hr>

Internal API
^^^^^^^^^^^^

.. autoclass:: skrl.envs.torch.wrappers.Wrapper
    :undoc-members:
    :show-inheritance:
    :members:
   
    .. automethod:: __init__

    .. py:property:: device

        The device used by the environment

        If the wrapped environment does not have the ``device`` property, the value of this property will be ``"cuda:0"`` or ``"cpu"`` depending on the device availability 

.. autoclass:: skrl.envs.torch.wrappers.OmniverseIsaacGymWrapper
    :undoc-members:
    :show-inheritance:
    :members:
   
    .. automethod:: __init__

.. autoclass:: skrl.envs.torch.wrappers.IsaacGymPreview3Wrapper
    :undoc-members:
    :show-inheritance:
    :members:
   
    .. automethod:: __init__

.. autoclass:: skrl.envs.torch.wrappers.IsaacGymPreview2Wrapper
    :undoc-members:
    :show-inheritance:
    :members:
   
    .. automethod:: __init__

.. autoclass:: skrl.envs.torch.wrappers.GymWrapper
    :undoc-members:
    :show-inheritance:
    :members:
   
    .. automethod:: __init__

.. autoclass:: skrl.envs.torch.wrappers.DeepMindWrapper
    :undoc-members:
    :show-inheritance:
    :private-members: _spec_to_space, _observation_to_tensor, _tensor_to_action
    :members:
   
    .. automethod:: __init__
