Parallel trainer
================

Concept
^^^^^^^

.. image:: ../_static/imgs/parallel_trainer.svg
    :width: 100%
    :align: center
    :alt: Parallel trainer

Basic usage
^^^^^^^^^^^

.. note::

    Each process adds a GPU memory overhead (~1GB, although it can be much higher) due to PyTorch's own CUDA kernels. See PyTorch `Issue #12873 <https://github.com/pytorch/pytorch/issues/12873>`_ for more details

.. code-block:: python
    :linenos:

    from skrl.trainers.torch import ConcurrentTrainer

    # asuming there is an environment called 'env'
    # and an agent or a list of agents called 'agents'

    # create a concurrent trainer
    cfg = {"timesteps": 50000, "headless": False}
    trainer = ConcurrentTrainer(cfg=cfg, env=env, agents=agents)
      
    # train the agent(s)
    trainer.start()

API
^^^

.. autoclass:: skrl.trainers.torch.parallel.ParallelTrainer
    :undoc-members:
    :show-inheritance:
    :inherited-members:
    :private-members: _pre_interaction, _post_interaction
    :members:

    .. automethod:: __init__
