Wrapping
========

This library works with a common API to interact with the RL environment

In order to operate with more complex or non-compatible interfaces and support interoperability between implementations a **wrapper mechanism is provided** which follows the following description:

Basic usage
^^^^^^^^^^^

.. tabs::
            
    .. tab:: Isaac Gym environment

        .. code-block:: python
            :linenos:

            # import the environment wrapper and loader
            from skrl.envs.torch import wrap_env
            from skrl.envs.torch import load_isaacgym_env_preview3

            # load the environment
            env = load_isaacgym_env_preview3(task_name="Cartpole")

            # wrap the environment
            env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview3")'

    .. tab:: OpenAI Gym environment
   
        .. code-block:: python
            :linenos:

            # import the environment wrapper and gym
            from skrl.envs.torch import wrap_env
            import gym

            # load environment
            env = gym.make('Pendulum-v1')

            # wrap the environment
            env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'

    .. tab:: DeepMind environment
   
        .. code-block:: python
            :linenos:

            # import the environment wrapper and the deepmind suite
            from skrl.envs.torch import wrap_env
            from dm_control import suite

            # load environment
            env = suite.load(domain_name="cartpole", task_name="swingup")

            # wrap the environment
            env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="dm")'

API
^^^

.. autofunction:: skrl.envs.torch.wrappers.wrap_env

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

.. autoclass:: skrl.envs.torch.wrappers.IsaacGymPreview2Wrapper
    :undoc-members:
    :show-inheritance:
    :members:
   
    .. automethod:: __init__

.. autoclass:: skrl.envs.torch.wrappers.IsaacGymPreview3Wrapper
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
