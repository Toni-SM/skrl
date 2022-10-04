.. _models_multivariate_gaussian:

Multivariate Gaussian model
===========================

Multivariate Gaussian models run **continuous-domain stochastic** policies.

skrl provides a Python mixin (:literal:`MultivariateGaussianMixin`) to assist in the creation of these types of models, allowing users to have full control over the function approximator definitions and architectures. Note that the use of this mixin must comply with the following rules:

* The definition of multiple inheritance must always include the :ref:`Model <models_base_class>` base class at the end.

  .. code-block:: python
      :emphasize-lines: 1

      class MultivariateGaussianModel(MultivariateGaussianMixin, Model):
          def __init__(self, observation_space, action_space, device="cuda:0",
                       clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
              Model.__init__(self, observation_space, action_space, device)
              MultivariateGaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

* The :ref:`Model <models_base_class>` base class constructor must be invoked before the mixins constructor.

  .. code-block:: python
      :emphasize-lines: 4-5

      class MultivariateGaussianModel(MultivariateGaussianMixin, Model):
          def __init__(self, observation_space, action_space, device="cuda:0",
                       clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
              Model.__init__(self, observation_space, action_space, device)
              MultivariateGaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

Concept
-------

.. image:: ../_static/imgs/model_multivariate_gaussian.svg
      :width: 100%
      :align: center
      :alt: Multivariate Gaussian model

Basic usage
-----------

.. tabs::

    .. tab:: Multi-Layer Perceptron (MLP)

        .. literalinclude:: ../snippets/multivariate_gaussian_model.py
            :language: python
            :linenos:
            :start-after: [start-mlp]
            :end-before: [end-mlp]

    .. tab:: Convolutional Neural Network (CNN)

        .. literalinclude:: ../snippets/multivariate_gaussian_model.py
            :language: python
            :linenos:
            :start-after: [start-cnn]
            :end-before: [end-cnn]

API
---

.. autoclass:: skrl.models.torch.multivariate_gaussian.MultivariateGaussianMixin
    :show-inheritance:
    :members:

    .. automethod:: __init__
