Models
======

.. toctree::
    :hidden:

    Tabular <models/tabular>
    Categorical <models/categorical>
    Gaussian <models/gaussian>
    Multivariate Gaussian <models/multivariate_gaussian>
    Deterministic <models/deterministic>
    Shared model <models/shared_model>

Models (or agent models) refer to a representation of the agent's policy, value function, etc. that the agent uses to make decisions. Agents can have one or more models, and their parameters are adjusted by the optimization algorithms.

.. raw:: html

    <br><hr>

Base class
----------

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.

.. raw:: html

    <br>

Mixin and inheritance
^^^^^^^^^^^^^^^^^^^^^

.. tabs::

    .. tab:: Mixin

        .. literalinclude:: ../snippets/model_mixin.py
            :language: python
            :start-after: [start-mixin]
            :end-before: [end-mixin]

    .. tab:: Model inheritance

        .. literalinclude:: ../snippets/model_mixin.py
            :language: python
            :start-after: [start-model]
            :end-before: [end-model]

.. raw:: html

    <br>

API
^^^

.. _models_base_class:

.. autoclass:: skrl.models.torch.base.Model
    :undoc-members:
    :show-inheritance:
    :private-members: _get_space_size
    :members:

    .. automethod:: __init__

    .. py:property:: device

        Device to be used for the computations

    .. py:property:: observation_space

        Observation/state space. It is a replica of the class constructor parameter of the same name

    .. py:property:: action_space

        Action space. It is a replica of the class constructor parameter of the same name

    .. py:property:: num_observations

        Number of elements in the observation/state space

    .. py:property:: num_actions

        Number of elements in the action space
