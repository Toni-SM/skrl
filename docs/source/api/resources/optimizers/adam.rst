:tocdepth: 4

Adam
====

An extension of the Stochastic Gradient Descent (SGD) algorithm that adaptively changes the learning rate
for each neural network parameter.

|br| |hr|

Usage
-----

The classes are not intended to be used directly by the user, but by agent implementations.

* For JAX, the class is the result of isolating the Optax optimizer that is mixed with the model parameters, as defined
  in the `Flax's TrainState <https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#train-state>`_ class.
* For Warp, the class is the result of reimplementing the optimizer to support CUDA graphs and gradient clipping.

|

API
---

|

JAX
^^^

.. automodule:: skrl.resources.optimizers.jax.adam
.. autosummary::
    :nosignatures:

    Adam

.. autoclass:: skrl.resources.optimizers.jax.adam.Adam
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __new__

|

Warp
^^^^

.. automodule:: skrl.resources.optimizers.warp.adam
.. autosummary::
    :nosignatures:

    Adam

.. autoclass:: skrl.resources.optimizers.warp.adam.Adam
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:
