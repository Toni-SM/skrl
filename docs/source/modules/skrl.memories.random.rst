Random memory
=============

Basic usage
^^^^^^^^^^^

   .. code-block:: python
      :linenos:

      import torch
      from skrl.memories.torch import RandomMemory

      # create a random memory object
      memory = RandomMemory(memory_size=1000, num_envs=1, replacement=False)

      # create tensors in memory
      memory.create_tensor(name="states", size=(64, 64, 3), dtype=torch.float32)
      memory.create_tensor(name="actions", size=(4,1), dtype=torch.float32)
      memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
      memory.create_tensor(name="next_states", size=(64, 64, 3), dtype=torch.float32)
      memory.create_tensor(name="dones", size=1, dtype=torch.bool)

      # add data to the memory
      for i in range(500):
          memory.add_samples(states=torch.rand(64, 64, 3).view(-1),
                             actions=torch.rand(4,1).view(-1),
                             rewards=torch.rand(1),
                             next_states=torch.rand(64, 64, 3).view(-1),
                             dones=torch.randint(2, size=(1,)).bool())

      # sample a batch of data from the memory
      batch = memory.sample(batch_size=32, names=["states", "actions", "rewards", "next_states", "dones"])

API
^^^

.. autoclass:: skrl.memories.torch.random.RandomMemory
   :undoc-members:
   :show-inheritance:
   :inherited-members:
   :members:

   .. automethod:: __init__
   .. automethod:: __len__
