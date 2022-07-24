SKRL - Reinforcement Learning library (|version|)
=================================================

**skrl** is an open-source modular library for Reinforcement Learning written in Python (using `PyTorch <https://pytorch.org/>`_) and designed with a focus on readability, simplicity, and transparency of algorithm implementation. In addition to supporting the `OpenAI Gym <https://www.gymlibrary.ml>`_ and `DeepMind <https://github.com/deepmind/dm_env>`_ environment interfaces, it allows loading and configuring `NVIDIA Isaac Gym <https://developer.nvidia.com/isaac-gym>`_ and `NVIDIA Omniverse Isaac Gym <https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_isaac_gym.html>`_ environments, enabling agents' simultaneous training by scopes (subsets of environments among all available environments), which may or may not share resources, in the same run

**Main features:**
    * Clean code
    * Modularity and reusability
    * Documented library, code and implementations
    * Support for OpenAI Gym (single and vectorized), DeepMind, NVIDIA Isaac Gym (preview 2, 3 and 4) and NVIDIA Omniverse Isaac Gym environments
    * Simultaneous learning by scopes in OpenAI Gym (vectorized), NVIDIA Isaac Gym and NVIDIA Omniverse Isaac Gym

.. warning::

    **skrl** is under **active continuous development**. Make sure you always have the latest version 

| **GitHub repository:** https://github.com/Toni-SM/skrl 
| **Questions or discussions:** https://github.com/Toni-SM/skrl/discussions 

**Citing skrl:** To cite this library (created at `Mondragon Unibertsitatea <https://www.mondragon.edu/en/home>`_) use the following reference to its `article <https://arxiv.org/abs/2202.03825>`_: *"skrl: Modular and Flexible Library for Reinforcement Learning"*

.. code-block:: bibtex

    @article{serrano2022skrl,
    title={skrl: Modular and Flexible Library for Reinforcement Learning},
    author={Serrano-Mu{\~n}oz, Antonio and Arana-Arexolaleiba, Nestor and Chrysostomou, Dimitrios and B{\o}gh, Simon},
    journal={arXiv preprint arXiv:2202.03825},
    year={2022}
    }

.. raw:: html

    <hr>

User guide
----------

.. toctree::
    :maxdepth: 2

    intro/installation
    intro/getting_started
    intro/examples
    intro/data

.. raw:: html

    <hr>

Library components (overview)
-----------------------------

Agents
^^^^^^

    Definition of reinforcement learning algorithms that compute an optimal policy. All agents inherit from one and only one :doc:`base class <modules/skrl.agents.base_class>` (that defines a uniform interface and provides for common functionalities) but which is not tied to the implementation details of the algorithms

    * :doc:`Advantage Actor Critic <modules/skrl.agents.a2c>` (**A2C**)
    * :doc:`Adversarial Motion Priors <modules/skrl.agents.amp>` (**AMP**)
    * :doc:`Cross-Entropy Method <modules/skrl.agents.cem>` (**CEM**)
    * :doc:`Deep Deterministic Policy Gradient <modules/skrl.agents.ddpg>` (**DDPG**)
    * :doc:`Double Deep Q-Network <modules/skrl.agents.ddqn>` (**DDQN**)
    * :doc:`Deep Q-Network <modules/skrl.agents.dqn>` (**DQN**)
    * :doc:`Proximal Policy Optimization <modules/skrl.agents.ppo>` (**PPO**)
    * :doc:`Q-learning <modules/skrl.agents.q_learning>` (**Q-learning**)
    * :doc:`Soft Actor-Critic <modules/skrl.agents.sac>` (**SAC**)
    * :doc:`State Action Reward State Action <modules/skrl.agents.sarsa>` (**SARSA**)
    * :doc:`Twin-Delayed DDPG <modules/skrl.agents.td3>` (**TD3**)
    * :doc:`Trust Region Policy Optimization <modules/skrl.agents.trpo>` (**TRPO**)

.. toctree::
    :maxdepth: 1
    :caption: Agents
    :hidden:

    modules/skrl.agents.base_class
    A2C <modules/skrl.agents.a2c>
    AMP <modules/skrl.agents.amp>
    CEM <modules/skrl.agents.cem>
    DDPG <modules/skrl.agents.ddpg>
    DDQN <modules/skrl.agents.ddqn>
    DQN <modules/skrl.agents.dqn>
    PPO <modules/skrl.agents.ppo>
    Q-learning <modules/skrl.agents.q_learning>
    SAC <modules/skrl.agents.sac>
    SARSA <modules/skrl.agents.sarsa>
    TD3 <modules/skrl.agents.td3>
    TRPO <modules/skrl.agents.trpo>

Environments
^^^^^^^^^^^^

    Definition of the Isaac Gym (preview 2, 3 and 4) and Omniverse Isaac Gym environment loaders, and wrappers for the OpenAI Gym, DeepMind, Isaac Gym and Omniverse Isaac Gym environments

    * :doc:`Wrapping <modules/skrl.envs.wrapping>` **OpenAI Gym**, **DeepMind**, **Isaac Gym** and **Omniverse Isaac Gym** environments
    * Loading :doc:`Isaac Gym environments <modules/skrl.envs.isaac_gym>`
    * Loading :doc:`Omniverse Isaac Gym environments <modules/skrl.envs.omniverse_isaac_gym>`

.. toctree::
    :maxdepth: 1
    :caption: Environments
    :hidden:

    modules/skrl.envs.wrapping
    modules/skrl.envs.isaac_gym
    modules/skrl.envs.omniverse_isaac_gym

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

Resources
^^^^^^^^^

    Definition of resources used by the agents during training and/or evaluation, such as exploration noises or learning rate schedulers

    **Noises:** Definition of the noises used by the agents during the exploration stage. All noises inherit from a :ref:`base class <base-class-noise>` that defines a uniform interface

        * :ref:`Gaussian <gaussian-noise>` noise
        * :ref:`Ornstein-Uhlenbeck <ornstein-uhlenbeck-noise>` noise

    **Learning rate schedulers:** Definition of learning rate schedulers. All schedulers inherit from the PyTorch :literal:`_LRScheduler` class (see `how to adjust learning rate <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_ in the PyTorch documentation for more details)

        * :ref:`KL Adaptive <kl-adaptive-scheduler>`

    **Preprocessors:** Definition of preprocessors

        * :ref:`Running standard scaler <running-standard-scaler-preprocessor>`

.. toctree::
    :maxdepth: 2
    :caption: Resources
    :hidden:
    
    modules/skrl.resources.noises
    modules/skrl.resources.schedulers
    modules/skrl.resources.preprocessors

Utils
^^^^^

    Definition of helper functions and classes

    * :doc:`Utilities <modules/skrl.utils.utilities>`, e.g. setting the random seed
    * :doc:`Model instantiators <modules/skrl.utils.model_instantiators>`
    * Memory and Tensorboard :doc:`file post-processing <modules/skrl.utils.postprocessing>`
    * :doc:`Isaac Gym utils <modules/skrl.utils.isaacgym_utils>`

.. toctree::
    :maxdepth: 1
    :caption: Utils
    :hidden:
        
    modules/skrl.utils.utilities
    modules/skrl.utils.model_instantiators
    modules/skrl.utils.postprocessing
    modules/skrl.utils.isaacgym_utils
