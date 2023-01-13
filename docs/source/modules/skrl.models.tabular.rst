.. _models_tabular:

Tabular model
=============

Tabular models run **discrete-domain deterministic/stochastic** policies.

skrl provides a Python mixin (:literal:`TabularMixin`) to assist in the creation of these types of models, allowing users to have full control over the table definitions. Note that the use of this mixin must comply with the following rules:

* The definition of multiple inheritance must always include the :ref:`Model <models_base_class>` base class at the end.

  .. code-block:: python
      :emphasize-lines: 1

      class TabularModel(TabularMixin, Model):
          def __init__(self, observation_space, action_space, device="cuda:0", num_envs=1):
              Model.__init__(self, observation_space, action_space, device)
              TabularMixin.__init__(self, num_envs)

* The :ref:`Model <models_base_class>` base class constructor must be invoked before the mixins constructor.

  .. code-block:: python
      :emphasize-lines: 3-4

      class TabularModel(TabularMixin, Model):
          def __init__(self, observation_space, action_space, device="cuda:0", num_envs=1):
              Model.__init__(self, observation_space, action_space, device)
              TabularMixin.__init__(self, num_envs)

Basic usage
-----------

.. tabs::

    .. tab:: :math:`\epsilon`-greedy policy

        .. literalinclude:: ../snippets/tabular_model.py
            :language: python
            :linenos:
            :start-after: [start-epsilon-greedy]
            :end-before: [end-epsilon-greedy]

API
---

.. autoclass:: skrl.models.torch.tabular.TabularMixin
    :show-inheritance:
    :exclude-members: to, state_dict, load_state_dict, load, save
    :members:

    .. automethod:: __init__
