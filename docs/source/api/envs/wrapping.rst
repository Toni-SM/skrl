:tocdepth: 3

Wrapping (single-agent)
=======================

.. raw:: html

    <br><hr>

This library works with a common API to interact with the following RL environments:

* OpenAI `Gym <https://www.gymlibrary.dev>`_ / Farama `Gymnasium <https://gymnasium.farama.org/>`_ (single and vectorized environments)
* `Farama Shimmy <https://shimmy.farama.org/>`_
* `DeepMind <https://github.com/deepmind/dm_env>`_
* `robosuite <https://robosuite.ai/>`_
* `NVIDIA Isaac Gym <https://developer.nvidia.com/isaac-gym>`_ (preview 2, 3 and 4)
* `NVIDIA Isaac Orbit <https://isaac-orbit.github.io/orbit/index.html>`_
* `NVIDIA Omniverse Isaac Gym <https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_gym_isaac_gym.html>`_

To operate with them and to support interoperability between these non-compatible interfaces, a **wrapping mechanism is provided** as shown in the diagram below

.. raw:: html

    <br>

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

                .. tabs::

                    .. group-tab:: |_4| |pytorch| |_4|

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [pytorch-start-omniverse-isaacgym]
                            :end-before: [pytorch-end-omniverse-isaacgym]

                    .. group-tab:: |_4| |jax| |_4|

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [jax-start-omniverse-isaacgym]
                            :end-before: [jax-end-omniverse-isaacgym]

            .. tab:: Multi-threaded environment

                .. tabs::

                    .. group-tab:: |_4| |pytorch| |_4|

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [pytorch-start-omniverse-isaacgym-mt]
                            :end-before: [pytorch-end-omniverse-isaacgym-mt]

                    .. group-tab:: |_4| |jax| |_4|

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [jax-start-omniverse-isaacgym-mt]
                            :end-before: [jax-end-omniverse-isaacgym-mt]

    .. tab:: Isaac Orbit

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/wrapping.py
                    :language: python
                    :start-after: [pytorch-start-isaac-orbit]
                    :end-before: [pytorch-end-isaac-orbit]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/wrapping.py
                    :language: python
                    :start-after: [jax-start-isaac-orbit]
                    :end-before: [jax-end-isaac-orbit]

    .. tab:: Isaac Gym

        .. tabs::

            .. tab:: Preview 4 (isaacgymenvs.make)

                .. tabs::

                    .. group-tab:: |_4| |pytorch| |_4|

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [pytorch-start-isaacgym-preview4-make]
                            :end-before: [pytorch-end-isaacgym-preview4-make]

                    .. group-tab:: |_4| |jax| |_4|

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [jax-start-isaacgym-preview4-make]
                            :end-before: [jax-end-isaacgym-preview4-make]

            .. tab:: Preview 4

                .. tabs::

                    .. group-tab:: |_4| |pytorch| |_4|

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [pytorch-start-isaacgym-preview4]
                            :end-before: [pytorch-end-isaacgym-preview4]

                    .. group-tab:: |_4| |jax| |_4|

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [jax-start-isaacgym-preview4]
                            :end-before: [jax-end-isaacgym-preview4]

            .. tab:: Preview 3

                .. tabs::

                    .. group-tab:: |_4| |pytorch| |_4|

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [pytorch-start-isaacgym-preview3]
                            :end-before: [pytorch-end-isaacgym-preview3]

                    .. group-tab:: |_4| |jax| |_4|

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [jax-start-isaacgym-preview3]
                            :end-before: [jax-end-isaacgym-preview3]

            .. tab:: Preview 2

                .. tabs::

                    .. group-tab:: |_4| |pytorch| |_4|

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [pytorch-start-isaacgym-preview2]
                            :end-before: [pytorch-end-isaacgym-preview2]

                    .. group-tab:: |_4| |jax| |_4|

                        .. literalinclude:: ../../snippets/wrapping.py
                            :language: python
                            :start-after: [jax-start-isaacgym-preview2]
                            :end-before: [jax-end-isaacgym-preview2]

    .. tab:: Gym / Gymnasium

        .. tabs::

            .. tab:: Gym

                .. tabs::

                    .. tab:: Single environment

                        .. tabs::

                            .. group-tab:: |_4| |pytorch| |_4|

                                .. literalinclude:: ../../snippets/wrapping.py
                                    :language: python
                                    :start-after: [pytorch-start-gym]
                                    :end-before: [pytorch-end-gym]

                            .. group-tab:: |_4| |jax| |_4|

                                .. literalinclude:: ../../snippets/wrapping.py
                                    :language: python
                                    :start-after: [jax-start-gym]
                                    :end-before: [jax-end-gym]

                    .. tab:: Vectorized environment

                        Visit the Gym documentation (`Vector <https://www.gymlibrary.dev/api/vector>`__) for more information about the creation and usage of vectorized environments

                        .. tabs::

                            .. group-tab:: |_4| |pytorch| |_4|

                                .. literalinclude:: ../../snippets/wrapping.py
                                    :language: python
                                    :start-after: [pytorch-start-gym-vectorized]
                                    :end-before: [pytorch-end-gym-vectorized]

                            .. group-tab:: |_4| |jax| |_4|

                                .. literalinclude:: ../../snippets/wrapping.py
                                    :language: python
                                    :start-after: [jax-start-gym-vectorized]
                                    :end-before: [jax-end-gym-vectorized]

            .. tab:: Gymnasium

                .. tabs::

                    .. tab:: Single environment

                        .. tabs::

                            .. group-tab:: |_4| |pytorch| |_4|

                                .. literalinclude:: ../../snippets/wrapping.py
                                    :language: python
                                    :start-after: [pytorch-start-gymnasium]
                                    :end-before: [pytorch-end-gymnasium]

                            .. group-tab:: |_4| |jax| |_4|

                                .. literalinclude:: ../../snippets/wrapping.py
                                    :language: python
                                    :start-after: [jax-start-gymnasium]
                                    :end-before: [jax-end-gymnasium]

                    .. tab:: Vectorized environment

                        Visit the Gymnasium documentation (`Vector <https://gymnasium.farama.org/api/vector>`__) for more information about the creation and usage of vectorized environments

                        .. tabs::

                            .. group-tab:: |_4| |pytorch| |_4|

                                .. literalinclude:: ../../snippets/wrapping.py
                                    :language: python
                                    :start-after: [pytorch-start-gymnasium-vectorized]
                                    :end-before: [pytorch-end-gymnasium-vectorized]

                            .. group-tab:: |_4| |jax| |_4|

                                .. literalinclude:: ../../snippets/wrapping.py
                                    :language: python
                                    :start-after: [jax-start-gymnasium-vectorized]
                                    :end-before: [jax-end-gymnasium-vectorized]

    .. tab:: Shimmy

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/wrapping.py
                    :language: python
                    :start-after: [pytorch-start-shimmy]
                    :end-before: [pytorch-end-shimmy]

            .. group-tab:: |_4| |jax| |_4|

                .. literalinclude:: ../../snippets/wrapping.py
                    :language: python
                    :start-after: [jax-start-shimmy]
                    :end-before: [jax-end-shimmy]

    .. tab:: DeepMind

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/wrapping.py
                    :language: python
                    :start-after: [pytorch-start-deepmind]
                    :end-before: [pytorch-end-deepmind]

            .. .. group-tab:: |_4| |jax| |_4|

            ..     .. literalinclude:: ../../snippets/wrapping.py
            ..         :language: python
            ..         :start-after: [jax-start-deepmind]
            ..         :end-before: [jax-end-deepmind]

    .. tab:: robosuite

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/wrapping.py
                    :language: python
                    :start-after: [pytorch-start-robosuite]
                    :end-before: [pytorch-end-robosuite]

            .. .. group-tab:: |_4| |jax| |_4|

            ..     .. literalinclude:: ../../snippets/wrapping.py
            ..         :language: python
            ..         :start-after: [jax-start-robosuite]
            ..         :end-before: [jax-end-robosuite]

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autofunction:: skrl.envs.wrappers.torch.wrap_env

.. raw:: html

    <br>

API (JAX)
---------

.. autofunction:: skrl.envs.wrappers.jax.wrap_env

.. raw:: html

    <br>

Internal API (PyTorch)
----------------------

.. autoclass:: skrl.envs.wrappers.torch.Wrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

    .. py:property:: device

        The device used by the environment

        If the wrapped environment does not have the ``device`` property, the value of this property will be ``"cuda:0"`` or ``"cpu"`` depending on the device availability

.. autoclass:: skrl.envs.wrappers.torch.OmniverseIsaacGymWrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.wrappers.torch.IsaacOrbitWrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.wrappers.torch.IsaacGymPreview3Wrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.wrappers.torch.IsaacGymPreview2Wrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.wrappers.torch.GymWrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.wrappers.torch.GymnasiumWrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.wrappers.torch.DeepMindWrapper
    :undoc-members:
    :show-inheritance:
    :private-members: _spec_to_space, _observation_to_tensor, _tensor_to_action
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.wrappers.torch.RobosuiteWrapper
    :undoc-members:
    :show-inheritance:
    :private-members: _spec_to_space, _observation_to_tensor, _tensor_to_action
    :members:

    .. automethod:: __init__

.. raw:: html

    <br>

Internal API (JAX)
------------------

.. autoclass:: skrl.envs.wrappers.jax.Wrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

    .. py:property:: device

        The device used by the environment

        If the wrapped environment does not have the ``device`` property, the value of this property will be ``"cuda"`` or ``"cpu"`` depending on the device availability

.. autoclass:: skrl.envs.wrappers.jax.OmniverseIsaacGymWrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.wrappers.jax.IsaacOrbitWrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.wrappers.jax.IsaacGymPreview3Wrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.wrappers.jax.IsaacGymPreview2Wrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__

.. autoclass:: skrl.envs.wrappers.jax.GymnasiumWrapper
    :undoc-members:
    :show-inheritance:
    :members:

    .. automethod:: __init__
