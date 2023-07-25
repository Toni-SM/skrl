Resources
=========

.. toctree::
    :hidden:

    Noises <resources/noises>
    Preprocessors <resources/preprocessors>
    Learning rate schedulers <resources/schedulers>
    Optimizers <resources/optimizers>

Resources groups a variety of components that may be used to improve the agents' performance.

.. raw:: html

    <br><hr>

Available resources are :doc:`noises <resources/noises>`, input :doc:`preprocessors <resources/preprocessors>`, learning rate :doc:`schedulers <resources/schedulers>` and :doc:`optimizers <resources/optimizers>` (this last one only for JAX).

.. list-table::
    :header-rows: 1

    * - Noises
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - :doc:`Gaussian <resources/noises/gaussian>` noise
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Ornstein-Uhlenbeck <resources/noises/ornstein_uhlenbeck>` noise |_2|
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

.. list-table::
    :header-rows: 1

    * - Preprocessors
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - :doc:`Running standard scaler <resources/preprocessors/running_standard_scaler>` |_4|
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

.. list-table::
    :header-rows: 1

    * - Learning rate schedulers
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - :doc:`KL Adaptive <resources/schedulers/kl_adaptive>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

.. list-table::
    :header-rows: 1

    * - Optimizers
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
    * - :doc:`Adam <resources/optimizers/adam>`\ |_5| |_5| |_5| |_5| |_5| |_5| |_3|
      - .. centered:: :math:`\scriptscriptstyle \texttt{PyTorch}`
      - .. centered:: :math:`\blacksquare`
