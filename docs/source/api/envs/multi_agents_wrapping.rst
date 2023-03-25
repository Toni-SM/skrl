:tocdepth: 3

Wrapping (multi-agents)
=======================

.. raw:: html

    <br><hr>

This library works with a common API to interact with the following RL environments:

* Farama `PettingZoo <https://pettingzoo.farama.org>`_ (parallel API)
* `Bi-DexHands <https://github.com/PKU-MARL/DexterousHands>`_

To operate with them and to support interoperability between these non-compatible interfaces, a **wrapping mechanism is provided** as shown in the diagram below

.. image:: ../../_static/imgs/multi_agent_wrapping-light.svg
    :width: 100%
    :align: center
    :class: only-light
    :alt: Environment wrapping

.. image:: ../../_static/imgs/multi_agent_wrapping-dark.svg
    :width: 100%
    :align: center
    :class: only-dark
    :alt: Environment wrapping

.. raw:: html

    <br>

Usage
-----

.. tabs::

    .. tab:: PettingZoo

        .. code-block:: python
            :linenos:

            # import the environment wrapper
            from skrl.envs.torch.wrappers import wrap_env

            # import a PettingZoo environment
            from pettingzoo.butterfly import pistonball_v6

            # load the environment
            env = pistonball_v6.parallel_env(continuous=False, max_cycles=125)

            # wrap the environment
            env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="pettingzoo")'

    .. tab:: Bi-DexHands

        .. code-block:: python
            :linenos:

            # import the environment wrapper and loader
            from skrl.envs.torch.wrappers import wrap_env
            from skrl.envs.torch.loaders import load_bidexhands_env

            # load the environment
            env = load_bidexhands_env(task_name="ShadowHandOver")

            # wrap the environment
            env = wrap_env(env, wrapper="bidexhands")

.. raw:: html

    <br>

API
---

.. autofunction:: skrl.envs.torch.wrappers.wrap_env
    :noindex:

.. raw:: html

    <br>

Internal API
------------

.. autoclass:: skrl.envs.torch.wrappers.MultiAgentEnvWrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

    .. py:property:: device

        The device used by the environment

        If the wrapped environment does not have the ``device`` property, the value of this property will be ``"cuda:0"`` or ``"cpu"`` depending on the device availability

    .. py:property:: possible_agents

        A list of all possible_agents the environment could generate

.. autoclass:: skrl.envs.torch.wrappers.BiDexHandsWrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.torch.wrappers.PettingZooWrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__
