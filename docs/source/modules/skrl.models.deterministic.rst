.. _models_deterministic:

Deterministic model
===================

Deterministic models run **continuous-domain deterministic** policies.

skrl provides a Python mixin (:literal:`DeterministicMixin`) to assist in the creation of these types of models, allowing users to have full control over the function approximator definitions and architectures. Note that the use of this mixin must comply with the following rules:

* The definition of multiple inheritance must always include the :ref:`Model <models_base_class>` base class at the end.

  .. code-block:: python
      :emphasize-lines: 1

      class DeterministicModel(DeterministicMixin, Model):
          def __init__(self, observation_space, action_space, device="cuda:0", clip_actions=False):
              Model.__init__(self, observation_space, action_space, device)
              DeterministicMixin.__init__(self, clip_actions)

* The :ref:`Model <models_base_class>` base class constructor must be invoked before the mixins constructor.

  .. code-block:: python
      :emphasize-lines: 3-4

      class DeterministicModel(DeterministicMixin, Model):
          def __init__(self, observation_space, action_space, device="cuda:0", clip_actions=False):
              Model.__init__(self, observation_space, action_space, device)
              DeterministicMixin.__init__(self, clip_actions)

Concept
-------

.. image:: ../_static/imgs/model_deterministic.svg
    :width: 65%
    :align: center
    :alt: Deterministic model

Basic usage
-----------

* Multi-Layer Perceptron (**MLP**)
* Convolutional Neural Network (**CNN**)

.. tabs::

    .. tab:: MLP

        .. image:: ../_static/imgs/model_deterministic_mlp.svg
            :width: 35%
            :align: center

        .. raw:: html

            <br>

        .. tabs::

            .. tab:: nn.Sequential

                .. literalinclude:: ../snippets/deterministic_model.py
                    :language: python
                    :linenos:
                    :start-after: [start-mlp-sequential]
                    :end-before: [end-mlp-sequential]

            .. tab:: nn.functional

                .. literalinclude:: ../snippets/deterministic_model.py
                    :language: python
                    :linenos:
                    :start-after: [start-mlp-functional]
                    :end-before: [end-mlp-functional]

    .. tab:: CNN

        .. image:: ../_static/imgs/model_deterministic_cnn.svg
            :width: 100%
            :align: center

        .. raw:: html

            <br>

        .. tabs::

            .. tab:: nn.Sequential

                .. literalinclude:: ../snippets/deterministic_model.py
                    :language: python
                    :linenos:
                    :start-after: [start-cnn-sequential]
                    :end-before: [end-cnn-sequential]

            .. tab:: nn.functional

                .. literalinclude:: ../snippets/deterministic_model.py
                    :language: python
                    :linenos:
                    :start-after: [start-cnn-functional]
                    :end-before: [end-cnn-functional]

API
---

.. autoclass:: skrl.models.torch.deterministic.DeterministicMixin
    :show-inheritance:
    :members:

    .. automethod:: __init__
