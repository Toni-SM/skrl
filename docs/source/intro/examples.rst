.. _examples:

Examples
========

.. contents:: Table of Contents
   :depth: 1
   :local:
   :backlinks: none

Learning in a Gym environment (one agent, one environment)
----------------------------------------------------------

This example performs the training of one agent in an OpenAI Gym environment. The following components or practices are exemplified (highlighted):

    - Load and wrap an OpenAI Gym environment: **Pendulum (DDPG)**
    - Instantiate models using the model instantiation utility: **CartPole (DQN)**
    - Load a checkpoint during evaluation: **Pendulum (DDPG)**, **CartPole (DQN)**

.. tabs::
            
    .. tab:: Pendulum (DDPG)

        .. tabs::
            
            .. tab:: Training

                View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/examples/gym_pendulum_ddpg.py>`_

                .. literalinclude:: ../examples/gym_pendulum_ddpg.py
                    :language: python
                    :linenos:
                    :emphasize-lines: 13, 49-55, 99

            .. tab:: Evaluation
                
                View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/examples/gym_pendulum_ddpg_eval.py>`_

                .. literalinclude:: ../examples/gym_pendulum_ddpg_eval.py
                    :language: python
                    :linenos:
                    :emphasize-lines: 45-48, 51

    .. tab:: CartPole (DQN)

        .. tabs::
            
            .. tab:: Training
                
                View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/examples/gym_cartpole_dqn.py>`_

                .. literalinclude:: ../examples/gym_cartpole_dqn.py
                    :language: python
                    :linenos:
                    :emphasize-lines: 4, 31-50, 69
        
            .. tab:: Evaluation
                
                View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/examples/gym_cartpole_dqn_eval.py>`_

                .. literalinclude:: ../examples/gym_cartpole_dqn_eval.py
                    :language: python
                    :linenos:
                    :emphasize-lines: 26-36, 39

Learning in an Isaac Gym environment (one agent, multiple environments)
-----------------------------------------------------------------------

This example performs the training of an agent in Isaac Gym's Cartpole environment. It tries to load the environment from preview 3, but if it fails, it will try to load the environment from preview 2. The following components or practices are exemplified (highlighted):

    - Load and wrap an Isaac Gym environment
    - Load a checkpoint during evaluation

.. tabs::
            
    .. tab:: Isaac Gym (one agent)

        .. tabs::
            
            .. tab:: Training
                
                View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/examples/isaacgym_cartpole_ppo.py>`_

                .. literalinclude:: ../examples/isaacgym_cartpole_ppo.py
                    :language: python
                    :linenos:
                    :emphasize-lines: 12-13, 53-58, 102

            .. tab:: Evaluation
                
                View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/examples/isaacgym_cartpole_ppo_eval.py>`_

                .. literalinclude:: ../examples/isaacgym_cartpole_ppo_eval.py
                    :language: python
                    :linenos:
                    :emphasize-lines: 49-50, 53

Learning by scopes in an Isaac Gym environment (parallel agents, multiple environments)
---------------------------------------------------------------------------------------

This example performs the training of 3 agents by scopes in Isaac Gym's Cartpole environment in the same run. It tries to load the environment from preview 3, but if it fails, it will try to load the environment from preview 2

.. image:: ../_static/imgs/example_parallel.jpg
      :width: 100%
      :align: center
      :alt: Simultaneous training

Two versions are presented:

    - Simultaneous training of agents **sharing the same memory** and whose scopes are automatically selected as equally as possible
    - Simultaneous training of agents **with individual memory** (no memory sharing) and whose scopes are manually specified and differ from each other

The following components or practices are exemplified (highlighted):

    - Create a shared memory: **Shared memory**
    - Learning by scopes (automatically defined): **Shared memory**
    - Create non-shared memories: **No shared memory**
    - Learning by scopes (manually defined): **No shared memory**
    - Load a checkpoint during evaluation: **Shared memory**, **No shared memory**

.. tabs::
            
    .. tab:: Shared memory

        .. tabs::
            
            .. tab:: Training
                
                View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/examples/isaacgym_parallel_shared_memory.py>`_

                .. literalinclude:: ../examples/isaacgym_parallel_shared_memory.py
                    :language: python
                    :linenos:
                    :emphasize-lines: 81, 152, 159, 166, 177-178

            .. tab:: Evaluation
                
                View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/examples/isaacgym_parallel_shared_memory_eval.py>`_

                .. literalinclude:: ../examples/isaacgym_parallel_shared_memory_eval.py
                    :language: python
                    :linenos:
                    :emphasize-lines: 64-67, 70-75, 78-82, 85-87

    .. tab:: No shared memory

        .. tabs::
            
            .. tab:: Training
                
                View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/examples/isaacgym_parallel_no_shared_memory.py>`_

                .. literalinclude:: ../examples/isaacgym_parallel_no_shared_memory.py
                    :language: python
                    :linenos:
                    :emphasize-lines: 81-83, 154, 161, 168, 179-180

            .. tab:: Evaluation
                
                View the raw code `here <https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/examples/isaacgym_parallel_no_shared_memory_eval.py>`_

                .. literalinclude:: ../examples/isaacgym_parallel_no_shared_memory_eval.py
                    :language: python
                    :linenos:
                    :emphasize-lines: 64-67, 70-75, 78-82, 85-87

Learning in the Isaac Sim (2021.2.1) environment (one agent, one environment)
-----------------------------------------------------------------------------

This example performs the training of an agent in Isaac Sim's JetBot environment

.. code-block:: bash

    mkdir /isaac-sim/standalone_examples/api/omni.isaac.jetbot/skrl_example 
    cd /isaac-sim/standalone_examples/api/omni.isaac.jetbot/skrl_example

    /isaac-sim/kit/python/bin/python3 -m pip install -e git+https://github.com/Toni-SM/skrl.git#egg=skrl

    wget https://raw.githubusercontent.com/Toni-SM/skrl/main/docs/source/examples/isaacsim_jetbot.py

    cp ../stable_baselines_example/env.py .

    /isaac-sim/python.sh isaacsim_jetbot.py
