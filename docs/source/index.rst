SKRL - Reinforcement Learning library (|version|)
=================================================

.. raw:: html

    <a href="https://pypi.org/project/skrl">
        <img alt="pypi" src="https://img.shields.io/pypi/v/skrl">
    </a>
    <a href="https://huggingface.co/skrl">
        <img alt="huggingface" src="https://img.shields.io/badge/%F0%9F%A4%97%20models-hugging%20face-F8D521">
    </a>
    <a href="https://github.com/Toni-SM/skrl/discussions">
        <img alt="discussions" src="https://img.shields.io/github/discussions/Toni-SM/skrl">
    </a>
    <br>
    <a href="https://github.com/Toni-SM/skrl/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/Toni-SM/skrl">
    </a>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://skrl.readthedocs.io">
        <img alt="docs" src="https://readthedocs.org/projects/skrl/badge/?version=latest">
    </a>
    <a href="https://github.com/Toni-SM/skrl/actions/workflows/python-test.yml">
        <img alt="pytest" src="https://github.com/Toni-SM/skrl/actions/workflows/python-test.yml/badge.svg">
    </a>
    <a href="https://github.com/Toni-SM/skrl/actions/workflows/pre-commit.yml">
        <img alt="pre-commit" src="https://github.com/Toni-SM/skrl/actions/workflows/pre-commit.yml/badge.svg">
    </a>
    <br><br>

**skrl** is an open-source library for Reinforcement Learning written in Python (on top of `PyTorch <https://pytorch.org/>`_ and `JAX <https://jax.readthedocs.io>`_) and designed with a focus on modularity, readability, simplicity and transparency of algorithm implementation. In addition to supporting the OpenAI `Gym <https://www.gymlibrary.dev>`_ / Farama `Gymnasium <https://gymnasium.farama.org/>`_, `DeepMind <https://github.com/deepmind/dm_env>`_ and other environment interfaces, it allows loading and configuring `NVIDIA Isaac Gym <https://developer.nvidia.com/isaac-gym>`_, `NVIDIA Isaac Orbit <https://isaac-orbit.github.io/orbit/index.html>`_ and `NVIDIA Omniverse Isaac Gym <https://docs.omniverse.nvidia.com/isaacsim/latest/tutorial_gym_isaac_gym.html>`_ environments, enabling agents' simultaneous training by scopes (subsets of environments among all available environments), which may or may not share resources, in the same run.

**Main features:**
    * PyTorch (|_1| |pytorch| |_1|) and JAX (|_1| |jax| |_1|)
    * Clean code
    * Modularity and reusability
    * Documented library, code and implementations
    * Support for Gym/Gymnasium (single and vectorized), DeepMind, NVIDIA Isaac Gym (preview 2, 3 and 4), NVIDIA Isaac Orbit, NVIDIA Omniverse Isaac Gym environments, among others
    * Simultaneous learning by scopes in Gym/Gymnasium (vectorized), NVIDIA Isaac Gym, NVIDIA Isaac Orbit and NVIDIA Omniverse Isaac Gym

.. raw:: html

    <br>

.. warning::

    **skrl** is under **active continuous development**. Make sure you always have the latest version. Visit the `develop <https://github.com/Toni-SM/skrl/tree/develop>`_ branch or its `documentation <https://skrl.readthedocs.io/en/develop>`_ to access the latest updates to be released.

| **GitHub repository:** https://github.com/Toni-SM/skrl
| **Questions or discussions:** https://github.com/Toni-SM/skrl/discussions
|

**Citing skrl:** To cite this library (created at Mondragon Unibertsitatea) use the following reference to its article: `skrl: Modular and Flexible Library for Reinforcement Learning <http://jmlr.org/papers/v24/23-0112.html>`_.

.. code-block:: bibtex

    @article{serrano2023skrl,
      author  = {Antonio Serrano-Muñoz and Dimitrios Chrysostomou and Simon Bøgh and Nestor Arana-Arexolaleiba},
      title   = {skrl: Modular and Flexible Library for Reinforcement Learning},
      journal = {Journal of Machine Learning Research},
      year    = {2023},
      volume  = {24},
      number  = {254},
      pages   = {1--9},
      url     = {http://jmlr.org/papers/v24/23-0112.html}
    }

.. raw:: html

    <br><hr>

User guide
----------

To start using the library, visit the following links:

.. toctree::
    :maxdepth: 1

    intro/installation
    intro/getting_started
    intro/examples
    intro/data

.. raw:: html

    <br><hr>

Library components (overview)
-----------------------------

.. toctree::
    :caption: API
    :hidden:

    api/agents
    api/multi_agents
    api/envs
    api/memories
    api/models
    api/resources
    api/trainers
    api/utils

Agents
^^^^^^

    Definition of reinforcement learning algorithms that compute an optimal policy. All agents inherit from one and only one :doc:`base class <api/agents>` (that defines a uniform interface and provides for common functionalities) but which is not tied to the implementation details of the algorithms

    * :doc:`Advantage Actor Critic <api/agents/a2c>` (**A2C**)
    * :doc:`Adversarial Motion Priors <api/agents/amp>` (**AMP**)
    * :doc:`Cross-Entropy Method <api/agents/cem>` (**CEM**)
    * :doc:`Deep Deterministic Policy Gradient <api/agents/ddpg>` (**DDPG**)
    * :doc:`Double Deep Q-Network <api/agents/ddqn>` (**DDQN**)
    * :doc:`Deep Q-Network <api/agents/dqn>` (**DQN**)
    * :doc:`Proximal Policy Optimization <api/agents/ppo>` (**PPO**)
    * :doc:`Q-learning <api/agents/q_learning>` (**Q-learning**)
    * :doc:`Robust Policy Optimization <api/agents/rpo>` (**RPO**)
    * :doc:`Soft Actor-Critic <api/agents/sac>` (**SAC**)
    * :doc:`State Action Reward State Action <api/agents/sarsa>` (**SARSA**)
    * :doc:`Twin-Delayed DDPG <api/agents/td3>` (**TD3**)
    * :doc:`Trust Region Policy Optimization <api/agents/trpo>` (**TRPO**)

Multi-agents
^^^^^^^^^^^^

    Definition of reinforcement learning algorithms that compute an optimal policies. All agents (multi-agents) inherit from one and only one :doc:`base class <api/multi_agents>` (that defines a uniform interface and provides for common functionalities) but which is not tied to the implementation details of the algorithms

    * :doc:`Independent Proximal Policy Optimization <api/multi_agents/ippo>` (**IPPO**)
    * :doc:`Multi-Agent Proximal Policy Optimization <api/multi_agents/mappo>` (**MAPPO**)

Environments
^^^^^^^^^^^^

    Definition of the Isaac Gym (preview 2, 3 and 4), Isaac Orbit and Omniverse Isaac Gym environment loaders, and wrappers for the Gym/Gymnasium, DeepMind, Isaac Gym, Isaac Orbit, Omniverse Isaac Gym environments, among others

    * :doc:`Single-agent environment wrapping <api/envs/wrapping>` for **Gym/Gymnasium**, **DeepMind**, **Isaac Gym**, **Isaac Orbit**, **Omniverse Isaac Gym** environments, among others
    * :doc:`Multi-agent environment wrapping <api/envs/multi_agents_wrapping>` for **PettingZoo** and **Bi-DexHands** environments
    * Loading :doc:`Isaac Gym environments <api/envs/isaac_gym>`
    * Loading :doc:`Isaac Orbit environments <api/envs/isaac_orbit>`
    * Loading :doc:`Omniverse Isaac Gym environments <api/envs/omniverse_isaac_gym>`

Memories
^^^^^^^^

    Generic memory definitions. Such memories are not bound to any agent and can be used for any role such as rollout buffer or experience replay memory, for example. All memories inherit from a :doc:`base class <api/memories>` that defines a uniform interface and keeps track (in allocated tensors) of transitions with the environment or other defined data

    * :doc:`Random memory <api/memories/random>`

Models
^^^^^^

    Definition of helper mixins for the construction of tabular functions or function approximators using artificial neural networks. This library does not provide predefined policies but helper mixins to create discrete and continuous (stochastic or deterministic) policies in which the user only has to define the tables (tensors) or artificial neural networks. All models inherit from one :doc:`base class <api/models>` that defines a uniform interface and provides for common functionalities. In addition, it is possible to create :doc:`shared model <api/models/shared_model>` by combining the implemented definitions

    * :doc:`Tabular model <api/models/tabular>` (discrete domain)
    * :doc:`Categorical model <api/models/categorical>` (discrete domain)
    * :doc:`Gaussian model <api/models/gaussian>` (continuous domain)
    * :doc:`Multivariate Gaussian model <api/models/multivariate_gaussian>` (continuous domain)
    * :doc:`Deterministic model <api/models/deterministic>` (continuous domain)

Trainers
^^^^^^^^

    Definition of the procedures responsible for managing the agent's training and interaction with the environment. All trainers inherit from a :doc:`base class <api/trainers>` that defines a uniform interface and provides for common functionalities

    * :doc:`Sequential trainer <api/trainers/sequential>`
    * :doc:`Parallel trainer <api/trainers/parallel>`
    * :doc:`Manual trainer <api/trainers/manual>`

Resources
^^^^^^^^^

    Definition of resources used by the agents during training and/or evaluation, such as exploration noises or learning rate schedulers

    **Noises:** Definition of the noises used by the agents during the exploration stage. All noises inherit from a :doc:`base class <api/resources/noises>` that defines a uniform interface

        * :doc:`Gaussian <api/resources/noises/gaussian>` noise
        * :doc:`Ornstein-Uhlenbeck <api/resources/noises/ornstein_uhlenbeck>` noise

    **Learning rate schedulers:** Definition of learning rate schedulers. All schedulers inherit from the PyTorch :literal:`_LRScheduler` class (see `how to adjust learning rate <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_ in the PyTorch documentation for more details)

        * :doc:`KL Adaptive <api/resources/schedulers/kl_adaptive>`

    **Preprocessors:** Definition of preprocessors

        * :doc:`Running standard scaler <api/resources/preprocessors/running_standard_scaler>`

    **Optimizers:** Definition of optimizers

        * :doc:`Adam <api/resources/optimizers/adam>`

Utils and configurations
^^^^^^^^^^^^^^^^^^^^^^^^

    Definition of utilities and configurations

    * :doc:`ML frameworks <api/config/frameworks>` configuration
    * :doc:`Random seed <api/utils/seed>`
    * Memory and Tensorboard :doc:`file post-processing <api/utils/postprocessing>`
    * :doc:`Model instantiators <api/utils/model_instantiators>`
    * :doc:`Hugging Face integration <api/utils/huggingface>`
    * :doc:`Isaac Gym utils <api/utils/isaacgym_utils>`
    * :doc:`Omniverse Isaac Gym utils <api/utils/omniverse_isaacgym_utils>`
