.. _models_base_class:

Base class
==========

.. note::

    This is the base class for all the other classes in this module.
    It provides the basic functionality for the other classes.
    **It is not intended to be used directly**.

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

API
^^^

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
