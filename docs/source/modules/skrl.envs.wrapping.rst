Wrapping
========

This library works with a common API to interact with the RL environment. 

In order to operate with complex or non-compatible interfaces and support interoperability between implementations a **wrapper mechanism is provided** which follows the following description:



Basic usage
^^^^^^^^^^^

* **Wrap an Isaac Gym environment:**

   .. code-block:: python
      :linenos:

      # import the environment wrapper and loader
      from skrl.env import wrap_env
      from skrl.utils.isaacgym_utils import load_isaacgym_env_preview3

      # load the environment
      env = load_isaacgym_env_preview3(task_name="Cartpole")

      # wrap the environment
      env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview3")'

* **Wrap an OpenAI Gym environment:**
   
   .. code-block:: python
      :linenos:

      # import the environment wrapper and gym
      from skrl.env import wrap_env
      import gym

      # load environment
      env = gym.make('Pendulum-v1')

      # wrap the environment
      env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'

API
^^^

.. autofunction:: skrl.env.wrapper.wrap_env

.. autoclass:: skrl.env.wrapper._Wrapper
   :undoc-members:
   :show-inheritance:
   :members:
   
   .. automethod:: __init__

.. autoclass:: skrl.env.wrapper._IsaacGymPreview2Wrapper
   :undoc-members:
   :show-inheritance:
   :members:
   
   .. automethod:: __init__

.. autoclass:: skrl.env.wrapper._IsaacGymPreview3Wrapper
   :undoc-members:
   :show-inheritance:
   :members:
   
   .. automethod:: __init__

.. autoclass:: skrl.env.wrapper._GymWrapper
   :undoc-members:
   :show-inheritance:
   :members:
   
   .. automethod:: __init__
