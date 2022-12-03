.. _models_categorical:

Categorical model
=================

Categorical models run **discrete-domain stochastic** policies.

skrl provides a Python mixin (:literal:`CategoricalMixin`) to assist in the creation of these types of models, allowing users to have full control over the function approximator definitions and architectures. Note that the use of this mixin must comply with the following rules:

* The definition of multiple inheritance must always include the :ref:`Model <models_base_class>` base class at the end.

  .. code-block:: python
      :emphasize-lines: 1

      class CategoricalModel(CategoricalMixin, Model):
          def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
              Model.__init__(self, observation_space, action_space, device)
              CategoricalMixin.__init__(self, unnormalized_log_prob)

* The :ref:`Model <models_base_class>` base class constructor must be invoked before the mixins constructor.

  .. code-block:: python
      :emphasize-lines: 3-4

      class CategoricalModel(CategoricalMixin, Model):
          def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
              Model.__init__(self, observation_space, action_space, device)
              CategoricalMixin.__init__(self, unnormalized_log_prob)

Concept
-------

.. image:: ../_static/imgs/model_categorical.svg
    :width: 100%
    :align: center
    :alt: Categorical model

Basic usage
-----------

* Multi-Layer Perceptron (**MLP**)
* Convolutional Neural Network (**CNN**)

.. tabs::

    .. tab:: MLP

        .. image:: ../_static/imgs/model_categorical_mlp.svg
            :width: 40%
            :align: center

        .. raw:: html

            <br>

        .. tabs::

            .. group-tab:: nn.Sequential

                .. literalinclude:: ../snippets/categorical_model.py
                    :language: python
                    :linenos:
                    :start-after: [start-mlp-sequential]
                    :end-before: [end-mlp-sequential]

            .. group-tab:: nn.functional

                .. literalinclude:: ../snippets/categorical_model.py
                    :language: python
                    :linenos:
                    :start-after: [start-mlp-functional]
                    :end-before: [end-mlp-functional]

    .. tab:: CNN

        .. image:: ../_static/imgs/model_categorical_cnn.svg
            :width: 100%
            :align: center

        .. raw:: html

            <br>

        .. tabs::

            .. group-tab:: nn.Sequential

                .. literalinclude:: ../snippets/categorical_model.py
                    :language: python
                    :linenos:
                    :start-after: [start-cnn-sequential]
                    :end-before: [end-cnn-sequential]

            .. group-tab:: nn.functional

                .. literalinclude:: ../snippets/categorical_model.py
                    :language: python
                    :linenos:
                    :start-after: [start-cnn-functional]
                    :end-before: [end-cnn-functional]

API
---

.. autoclass:: skrl.models.torch.categorical.CategoricalMixin
    :show-inheritance:
    :members:

    .. automethod:: __init__
