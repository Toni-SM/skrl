Resources
=========

.. toctree::
    :hidden:

    Noises <resources/noises>
    Preprocessors <resources/preprocessors>
    Learning rate schedulers <resources/schedulers>
    Optimizers <resources/optimizers>

Resources groups a variety of components that may be used to improve the agents' performance.

|br| |hr|

Resources are grouped into four categories:

* :doc:`Noises <resources/noises>`
* Input :doc:`preprocessors <resources/preprocessors>`
* Learning rate :doc:`schedulers <resources/schedulers>`
* :doc:`Optimizers <resources/optimizers>`

Implemented resources
---------------------

The following table lists the implemented resources and their support for different frameworks.

.. list-table::
    :header-rows: 1

    * - Noises
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`Gaussian <resources/noises/gaussian>` noise
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
    * - :doc:`Ornstein-Uhlenbeck <resources/noises/ornstein_uhlenbeck>` noise |_2|
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

.. list-table::
    :header-rows: 1

    * - Preprocessors
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`Running standard scaler <resources/preprocessors/running_standard_scaler>` |_4|
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

.. list-table::
    :header-rows: 1

    * - Learning rate schedulers
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`KL Adaptive <resources/schedulers/kl_adaptive>`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`

.. list-table::
    :header-rows: 1

    * - Optimizers
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`Adam <resources/optimizers/adam>`\ |_5| |_5| |_5| |_5| |_5| |_5| |_3|
      - .. centered:: :math:`\scriptscriptstyle \texttt{PyTorch}`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\scriptscriptstyle \texttt{Warp-NN}`
