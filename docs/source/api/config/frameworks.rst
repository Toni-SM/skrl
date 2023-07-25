ML frameworks configuration
===========================

Configurations for behavior modification of Machine Learning (ML) frameworks.

.. raw:: html

    <br><hr>

JAX
---

JAX specific configuration

.. raw:: html

    <br>

API
^^^

.. py:data:: skrl.config.jax.backend
    :type: str
    :value: "numpy"

    Backend used by the different components to operate and generate arrays

    This configuration excludes models and optimizers.
    Supported backend are: ``"numpy"`` and ``"jax"``

.. py:data:: skrl.config.jax.key
    :type: jnp.ndarray
    :value: [0, 0]

    Pseudo-random number generator (PRNG) key
