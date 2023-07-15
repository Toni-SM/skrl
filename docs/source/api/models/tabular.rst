.. _models_tabular:

Tabular model
=============

Tabular models run **discrete-domain deterministic/stochastic** policies.

.. raw:: html

    <br><hr>

skrl provides a Python mixin (:literal:`TabularMixin`) to assist in the creation of these types of models, allowing users to have full control over the table definitions. Note that the use of this mixin must comply with the following rules:

* The definition of multiple inheritance must always include the :ref:`Model <models_base_class>` base class at the end.

* The :ref:`Model <models_base_class>` base class constructor must be invoked before the mixins constructor.

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. literalinclude:: ../../snippets/tabular_model.py
            :language: python
            :emphasize-lines: 1, 3-4
            :start-after: [start-definition-torch]
            :end-before: [end-definition-torch]

.. raw:: html

    <br>

Usage
-----

.. tabs::

    .. tab:: :math:`\epsilon`-greedy policy

        .. tabs::

            .. group-tab:: |_4| |pytorch| |_4|

                .. literalinclude:: ../../snippets/tabular_model.py
                    :language: python
                    :start-after: [start-epsilon-greedy-torch]
                    :end-before: [end-epsilon-greedy-torch]

.. raw:: html

    <br>

API (PyTorch)
-------------

.. autoclass:: skrl.models.torch.tabular.TabularMixin
    :show-inheritance:
    :exclude-members: to, state_dict, load_state_dict, load, save
    :members:

    .. automethod:: __init__
