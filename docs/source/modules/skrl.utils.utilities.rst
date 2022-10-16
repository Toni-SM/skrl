Utilities
=========

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

.. raw:: html

   <hr>

Seed the random number generators
---------------------------------

API
"""

.. autofunction:: skrl.utils.set_seed

Weights and Biases
------------------

Integration
"""""""""""

You can use `Weights & Biases <https://wandb.ai>`_ to easily track your experiments.
Please login to your account and create a new project.
Afterwards, log into Weights & Biases from your terminal:

.. code-block:: bash

   wandb login

Usage
"""""

Change your agents config to enable Weights & Biases, also, at a minimum, enter your project name and entity name:

.. code-block:: python

   from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG

   cfg_dqn = DQN_DEFAULT_CONFIG.copy()

   cfg_dqn["experiment"]["wandb"] = {
         "enabled": True,
         "project": "skrl",
         "entity": "Toni-SM",
      }
