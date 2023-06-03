:tocdepth: 4

Wrapping (single-agent)
=======================

.. raw:: html

    <br><hr>

This library works with a common API to interact with the following RL environments:

* OpenAI `Gym <https://www.gymlibrary.dev>`_ / Farama `Gymnasium <https://gymnasium.farama.org/>`_ (single and vectorized environments)
* `DeepMind <https://github.com/deepmind/dm_env>`_
* `robosuite <https://robosuite.ai/>`_
* `NVIDIA Isaac Gym <https://developer.nvidia.com/isaac-gym>`_ (preview 2, 3 and 4)
* `NVIDIA Isaac Orbit <https://isaac-orbit.github.io/orbit/index.html>`_
* `NVIDIA Omniverse Isaac Gym <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_isaac_gym.html>`_

To operate with them and to support interoperability between these non-compatible interfaces, a **wrapping mechanism is provided** as shown in the diagram below

.. image:: ../../_static/imgs/wrapping-light.svg
    :width: 100%
    :align: center
    :class: only-light
    :alt: Environment wrapping

.. image:: ../../_static/imgs/wrapping-dark.svg
    :width: 100%
    :align: center
    :class: only-dark
    :alt: Environment wrapping

.. raw:: html

    <br>

Usage
-----

.. tabs::

    .. tab:: Omniverse Isaac Gym

        .. tabs::

            .. tab:: Common environment

                .. literalinclude:: ../../snippets/wrapping.py
                    :language: python
                    :start-after: [pytorch-start-omniverse-isaacgym]
                    :end-before: [pytorch-end-omniverse-isaacgym]

            .. tab:: Multi-threaded environment

                .. literalinclude:: ../../snippets/wrapping.py
                    :language: python
                    :start-after: [pytorch-start-omniverse-isaacgym-mt]
                    :end-before: [pytorch-end-omniverse-isaacgym-mt]

    .. tab:: Isaac Orbit

        .. literalinclude:: ../../snippets/wrapping.py
            :language: python
            :start-after: [pytorch-start-isaac-orbit]
            :end-before: [pytorch-end-isaac-orbit]

    .. tab:: Isaac Gym

        .. tabs::

            .. tab:: Preview 4 (isaacgymenvs.make)

                .. literalinclude:: ../../snippets/wrapping.py
                    :language: python
                    :start-after: [pytorch-start-isaacgym-preview4-make]
                    :end-before: [pytorch-end-isaacgym-preview4-make]

            .. tab:: Preview 4

                .. literalinclude:: ../../snippets/wrapping.py
                    :language: python
                    :start-after: [pytorch-start-isaacgym-preview4]
                    :end-before: [pytorch-end-isaacgym-preview4]

            .. tab:: Preview 3

                .. literalinclude:: ../../snippets/wrapping.py
                    :language: python
                    :start-after: [pytorch-start-isaacgym-preview3]
                    :end-before: [pytorch-end-isaacgym-preview3]

            .. tab:: Preview 2

                .. literalinclude:: ../../snippets/wrapping.py
                    :language: python
                    :start-after: [pytorch-start-isaacgym-preview2]
                    :end-before: [pytorch-end-isaacgym-preview2]

    .. tab:: Gym / Gymnasium

        .. tabs::

            .. tab:: Gym

                .. tabs::

                    .. tab:: Single environment

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [pytorch-start-gym]
                            :end-before: [pytorch-end-gym]

                    .. tab:: Vectorized environment

                        Visit the Gym documentation (`Vector <https://www.gymlibrary.dev/api/vector>`__) for more information about the creation and usage of vectorized environments

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [pytorch-start-gym-vectorized]
                            :end-before: [pytorch-end-gym-vectorized]

            .. tab:: Gymnasium

                .. tabs::

                    .. tab:: Single environment

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [pytorch-start-gymnasium]
                            :end-before: [pytorch-end-gymnasium]

                    .. tab:: Vectorized environment

                        Visit the Gymnasium documentation (`Vector <https://gymnasium.farama.org/api/vector>`__) for more information about the creation and usage of vectorized environments

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [pytorch-start-gymnasium-vectorized]
                            :end-before: [pytorch-end-gymnasium-vectorized]

    .. tab:: DeepMind

        .. literalinclude:: ../../snippets/wrapping.py
            :language: python
            :start-after: [pytorch-start-deepmind]
            :end-before: [pytorch-end-deepmind]

    .. tab:: robosuite

        .. literalinclude:: ../../snippets/wrapping.py
            :language: python
            :start-after: [pytorch-start-robosuite]
            :end-before: [pytorch-end-robosuite]

.. raw:: html

    <br>

API
---

.. autofunction:: skrl.envs.torch.wrappers.wrap_env

.. raw:: html

    <br>

Internal API
------------

PyTorch
"""""""

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

.. autoclass:: skrl.envs.torch.wrappers.IsaacOrbitWrapper
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

.. autoclass:: skrl.envs.torch.wrappers.GymnasiumWrapper
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

.. autoclass:: skrl.envs.torch.wrappers.RobosuiteWrapper
    :undoc-members:
    :show-inheritance:
    :private-members: _spec_to_space, _observation_to_tensor, _tensor_to_action
    :members:

    .. automethod:: __init__

JAX
"""

.. autoclass:: skrl.envs.jax.wrappers.OmniverseIsaacGymWrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.jax.wrappers.IsaacGymPreview3Wrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.jax.wrappers.IsaacGymPreview2Wrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.jax.wrappers.GymnasiumWrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__
