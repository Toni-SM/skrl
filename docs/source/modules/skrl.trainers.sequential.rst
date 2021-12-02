Sequential trainer
==================

Basic usage
^^^^^^^^^^^

   .. code-block:: python
      :linenos:
 
      from skrl.trainers.torch import SequentialTrainer

      # asuming there is an environment called 'env'
      # and an agent or a list of agents called 'agents'

      # create a sequential trainer
      cfg = {"timesteps": 50000, "headless": False}
      trainer = SequentialTrainer(cfg=cfg, env=env, agents=agents)
      
      # train the agent(s)
      trainer.start()

API
^^^

.. autoclass:: skrl.trainers.torch.sequential.SequentialTrainer
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :private-members: _pre_interaction, _post_interaction
   :members:
   
   .. automethod:: __init__

