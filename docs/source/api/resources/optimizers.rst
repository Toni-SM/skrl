:tocdepth: 4

.. _optimizers:

Optimizers
==========

.. toctree::
    :hidden:

    Adam <optimizers/adam>

Optimizers are algorithms that adjust the parameters of artificial neural networks
to minimize the error or loss function during the training process.

|br| |hr|

Implemented optimizers
----------------------

The following table lists the implemented optimizers and their support for different frameworks.

.. list-table::
    :header-rows: 1

    * - Optimizers
      - .. centered:: |_4| |pytorch| |_4|
      - .. centered:: |_4| |jax| |_4|
      - .. centered:: |_4| |warp| |_4|
    * - :doc:`Adam <optimizers/adam>`\ |_5| |_5| |_5| |_5| |_5| |_5| |_3|
      - .. centered:: :math:`\scriptscriptstyle \texttt{PyTorch}`
      - .. centered:: :math:`\blacksquare`
      - .. centered:: :math:`\blacksquare`
