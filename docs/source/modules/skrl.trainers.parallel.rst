Parallel trainer
================

Basic usage
^^^^^^^^^^^

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
