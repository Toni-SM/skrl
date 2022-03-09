SKRL - Reinforcement Learning library
=====================================

**skrl** is an open-source modular library for Reinforcement Learning written in Python (using `PyTorch <https://pytorch.org/>`_) and designed with a focus on readability, simplicity, and transparency of algorithm implementation. In addition to supporting the `Gym <https://gym.openai.com/>`_ interface, it allows loading and configuring `NVIDIA Isaac Gym <https://developer.nvidia.com/isaac-gym>`_ environments, enabling agents' simultaneous training by scopes (subsets of environments among all available environments), which may or may not share resources, in the same run

**Main features:**

    * Clean code
    * Modularity and reusability
    * Documented library, code and implementations
    * Isaac Gym environment loading (preview 2 and 3)
    * Simultaneous learning by scopes in Isaac Gym environments

.. warning::

    **skrl** is under **active continuous development**. Make sure you always have the latest version 

**GitHub repository:** https://github.com/Toni-SM/skrl 

**Citing skrl:** To cite this library (created at `Mondragon Unibertsitatea <https://www.mondragon.edu/en/home>`_) use the following reference to its `article <https://arxiv.org/abs/2202.03825>`_: *"skrl: Modular and Flexible Library for Reinforcement Learning"*

.. code-block:: bibtex

    @article{serrano2022skrl,
    title={skrl: Modular and Flexible Library for Reinforcement Learning},
    author={Serrano-Mu{\~n}oz, Antonio and Arana-Arexolaleiba, Nestor and Chrysostomou, Dimitrios and B{\o}gh, Simon},
    journal={arXiv preprint arXiv:2202.03825},
    year={2022}
    }

User guide
----------

.. toctree::
    :maxdepth: 2

    intro/installation
    intro/examples
    intro/data

------------

Library components (overview)
-----------------------------

Agents
^^^^^^

    Definition of reinforcement learning algorithms that compute an optimal policy. All agents inherit from one and only one :doc:`base class <modules/skrl.agents.base_class>` (that defines a uniform interface and provides for common functionalities) but which is not tied to the implementation details of the algorithms

    * :doc:`Cross-Entropy Method <modules/skrl.agents.cem>` (**CEM**)
    * :doc:`Deep Deterministic Policy Gradient <modules/skrl.agents.ddpg>` (**DDPG**)
    * :doc:`Double Deep Q-Network <modules/skrl.agents.ddqn>` (**DDQN**)
    * :doc:`Deep Q-Network <modules/skrl.agents.dqn>` (**DQN**)
    * :doc:`Proximal Policy Optimization <modules/skrl.agents.ppo>` (**PPO**)
    * :doc:`Q-learning <modules/skrl.agents.q_learning>` (**Q-learning**)
    * :doc:`Soft Actor-Critic <modules/skrl.agents.sac>` (**SAC**)
    * :doc:`State Action Reward State Action <modules/skrl.agents.sarsa>` (**SARSA**)
    * :doc:`Twin-Delayed DDPG <modules/skrl.agents.td3>` (**TD3**)

.. toctree::
    :maxdepth: 1
    :caption: Agents
    :hidden:

    modules/skrl.agents.base_class
    CEM <modules/skrl.agents.cem>
    DDPG <modules/skrl.agents.ddpg>
    DDQN <modules/skrl.agents.ddqn>
    DQN <modules/skrl.agents.dqn>
    PPO <modules/skrl.agents.ppo>
    Q-learning <modules/skrl.agents.q_learning>
    SAC <modules/skrl.agents.sac>
    SARSA <modules/skrl.agents.sarsa>
    TD3 <modules/skrl.agents.td3>

Environments
^^^^^^^^^^^^

    Definition of the Isaac Gym environment loaders (preview 2 and preview 3) and wrappers for the OpenAI Gym and Isaac Gym environments

    * :doc:`Wrapping <modules/skrl.envs.wrapping>`
    * :doc:`Isaac Gym environments <modules/skrl.envs.isaac_gym>`

.. toctree::
    :maxdepth: 1
    :caption: Environments
    :hidden:

    modules/skrl.envs.wrapping
    modules/skrl.envs.isaac_gym

Memories
^^^^^^^^

    Generic memory definitions. Such memories are not bound to any agent and can be used for any role such as rollout buffer or experience replay memory, for example. All memories inherit from a :doc:`base class <modules/skrl.memories.base_class>` that defines a uniform interface and keeps track (in allocated tensors) of transitions with the environment or other defined data

    * :doc:`Random memory <modules/skrl.memories.random>`

.. toctree::
    :maxdepth: 1
    :caption: Memories
    :hidden:

    modules/skrl.memories.base_class
    modules/skrl.memories.random
    .. modules/skrl.memories.prioritized

Models
^^^^^^

    Definition of helper classes for the construction of tabular functions or function approximators using artificial neural networks. This library does not provide predefined policies but helper classes to create discrete and continuous (stochastic or deterministic) policies in which the user only has to define the tables (tensors) or artificial neural networks. All models inherit from one :doc:`base class <modules/skrl.models.base_class>` that defines a uniform interface and provides for common functionalities

    * :doc:`Tabular model <modules/skrl.models.tabular>` (discrete domain)
    * :doc:`Categorical model <modules/skrl.models.categorical>` (discrete domain)
    * :doc:`Gaussian model <modules/skrl.models.gaussian>` (continuous domain)
    * :doc:`Deterministic model <modules/skrl.models.deterministic>` (continuous domain)

.. toctree::
    :maxdepth: 1
    :caption: Models
    :hidden:

    modules/skrl.models.base_class
    modules/skrl.models.tabular
    modules/skrl.models.categorical
    modules/skrl.models.gaussian
    modules/skrl.models.deterministic

Noises
^^^^^^

    Definition of the noises used by the agents during the exploration stage. All noises inherit from a :doc:`base class <modules/skrl.noises.base_class>` that defines a uniform interface

    * :doc:`Gaussian <modules/skrl.noises.gaussian>` noise
    * :doc:`Ornstein-Uhlenbeck <modules/skrl.noises.ornstein_uhlenbeck>` noise

.. toctree::
    :maxdepth: 1
    :caption: Noises
    :hidden:
        
    modules/skrl.noises.base_class
    modules/skrl.noises.gaussian
    modules/skrl.noises.ornstein_uhlenbeck

Trainers
^^^^^^^^

    Definition of the procedures responsible for managing the agent's training and interaction with the environment. All trainers inherit from a :doc:`base class <modules/skrl.trainers.base_class>` that defines a uniform interface and provides for common functionalities

    * :doc:`Sequential trainer <modules/skrl.trainers.sequential>`
    * :doc:`Parallel trainer <modules/skrl.trainers.parallel>`

.. toctree::
    :maxdepth: 1
    :caption: Trainers
    :hidden:
        
    modules/skrl.trainers.base_class
    modules/skrl.trainers.sequential
    modules/skrl.trainers.parallel

Utils
^^^^^

    Definition of helper functions and classes

    * :doc:`Model instantiators <modules/skrl.utils.model_instantiators>`
    * :doc:`File post-processing <modules/skrl.utils.postprocessing>`
    * :doc:`Isaac Gym utils <modules/skrl.utils.isaacgym_utils>`

.. toctree::
    :maxdepth: 1
    :caption: Utils
    :hidden:
        
    modules/skrl.utils.model_instantiators
    modules/skrl.utils.postprocessing
    modules/skrl.utils.isaacgym_utils
