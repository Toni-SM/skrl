Adam
====

An extension of the stochastic gradient descent algorithm that adaptively changes the learning rate for each neural network parameter.

.. raw:: html

    <br><hr>

Usage
-----

.. note::

    This class is the result of isolating the Optax optimizer that is mixed with the model parameters, as defined in the `Flax's TrainState <https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#train-state>`_ class. It is not intended to be used directly by the user, but by agent implementations.

.. tabs::

    .. group-tab:: |_4| |jax| |_4|

        .. code-block:: python
            :emphasize-lines: 2, 5, 8

            # import the optimizer class
            from skrl.resources.optimizers.jax import Adam

            # instantiate the optimizer
            optimizer = Adam(model=model, lr=1e-3)

            # step the optimizer
            optimizer = optimizer.step(grad, model)

.. raw:: html

    <br>

API (JAX)
---------

.. autoclass:: skrl.resources.optimizers.jax.adam.Adam
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __new__
