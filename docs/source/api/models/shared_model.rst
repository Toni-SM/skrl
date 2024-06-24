Shared model
============

Sometimes it is desirable to define models that use shared layers or network to represent multiple function approximators. This practice, known as *shared parameters* (or *parameter sharing*), *shared layers*, *shared model*, *shared networks* or *joint architecture* among others, is typically justified by the following criteria:

* Learning the same characteristics, especially when processing large inputs (such as images, e.g.).

* Reduce the number of parameters in the whole system.

* Make the computation more efficient (single forward-pass).

.. raw:: html

    <br><hr>

Implementation
--------------

By combining the implemented mixins, it is possible to define shared models with skrl. In these cases, the use of the :literal:`role` argument (a Python string) is relevant. The agents will call the models by setting the :literal:`role` argument according to their requirements. Visit each agent's documentation (*Key* column of the table under *Spaces and models* section) to know the possible values that this parameter can take.

The code snippet below shows how to define a shared model. The following practices for building shared models can be identified:

* The definition of multiple inheritance must always include the :ref:`Model <models_base_class>` base class at the end.

* The :ref:`Model <models_base_class>` base class constructor must be invoked before the mixins constructor.

* All mixin constructors must be invoked.

  * Specify :literal:`role` argument is optional if all constructors belong to different mixins.

  * If multiple models of the same mixin type are required, the same constructor must be invoked as many times as needed. To do so, it is mandatory to specify the :literal:`role` argument.

* The :literal:`.act(...)` method needs to be overridden to disambiguate its call.

* The same instance of the shared model must be passed to all keys involved.

.. raw:: html

    <br>

.. tabs::

    .. group-tab:: |_4| |pytorch| |_4|

        .. tabs::

            .. group-tab:: Single forward-pass

                .. warning::

                    The implementation described for single forward-pass requires that the value-pass always follows the policy-pass (e.g.: ``PPO``) which may not be generalized to other algorithms.

                    If this requirement is not met, other forms of "chaching" the shared layers/network output could be implemented.

                .. literalinclude:: ../../snippets/shared_model.py
                    :language: python
                    :start-after: [start-mlp-single-forward-pass-torch]
                    :end-before: [end-mlp-single-forward-pass-torch]

            .. group-tab:: Multiple forward-pass

                .. literalinclude:: ../../snippets/shared_model.py
                    :language: python
                    :start-after: [start-mlp-multi-forward-pass-torch]
                    :end-before: [end-mlp-multi-forward-pass-torch]
