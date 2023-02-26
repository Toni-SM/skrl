Parallel trainer
================

Train agents in parallel using multiple processes.

.. raw:: html

    <br><hr>

Concept
-------

.. image:: ../../_static/imgs/parallel_trainer-light.svg
    :width: 100%
    :align: center
    :class: only-light
    :alt: Parallel trainer

.. image:: ../../_static/imgs/parallel_trainer-dark.svg
    :width: 100%
    :align: center
    :class: only-dark
    :alt: Parallel trainer

.. raw:: html

    <br>

Usage
-----

.. note::

    Each process adds a GPU memory overhead (~1GB, although it can be much higher) due to PyTorch's CUDA kernels. See PyTorch `Issue #12873 <https://github.com/pytorch/pytorch/issues/12873>`_ for more details

.. note::

    At the moment, only simultaneous training and evaluation of agents with local memory (no memory sharing) is implemented

.. tabs::

    .. tab:: Snippet

        .. literalinclude:: ../../snippets/trainer.py
            :language: python
            :linenos:
            :start-after: [start-parallel]
            :end-before: [end-parallel]

.. raw:: html

    <br>

Configuration
-------------

.. literalinclude:: ../../../../skrl/trainers/torch/parallel.py
    :language: python
    :lines: 15-19
    :linenos:

.. raw:: html

    <br>

API
---

.. autoclass:: skrl.trainers.torch.parallel.PARALLEL_TRAINER_DEFAULT_CONFIG

.. autoclass:: skrl.trainers.torch.parallel.ParallelTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :members:

    .. automethod:: __init__
