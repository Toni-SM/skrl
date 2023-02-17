Getting Started
===============

In this section, you will learn how to use the various components of the **skrl** library to create reinforcement learning tasks. Whether you are a beginner or an experienced researcher, we hope this section will provide you with a solid foundation to build upon. We recommend visiting the :ref:`Examples <examples>` to see how the components can be integrated and applied in practice. Let's get started!

Reinforcement Learning schema
-----------------------------

**Reinforcement Learning (RL)** is a Machine Learning sub-field for decision making that allows an agent to learn from its interaction with the environment as shown in the following schema:

.. image:: ../_static/imgs/rl_schema-light.svg
    :width: 100%
    :align: center
    :class: only-light
    :alt: Reinforcement Learning schema

.. image:: ../_static/imgs/rl_schema-dark.svg
    :width: 100%
    :align: center
    :class: only-dark
    :alt: Reinforcement Learning schema

.. raw:: html

    <br>

At each step (also called timestep) of interaction with the environment, the agent sees an observation :math:`o_t` of the complete description of the state :math:`s_t \in S` of the environment. Then, it decides which action :math:`a_t \in A` to take from the action space using a policy. The environment, which changes in response to the agent's action (or by itself), returns a reward signal :math:`r_t = R(s_t, a_t, s_{t+1})` as a measure of how good or bad the action was that moved it to its new state :math:`s_{t+1}`. The agent aims to maximize the cumulative reward (discounted or not by a factor :math:`\gamma \in (0,1]`) by adjusting the policy's behaviour via some optimization algorithm.

**From this schema, this section is intended to guide in the creation of a RL system using skrl**

1. Environments
---------------

The environment plays a fundamental role in the definition of the RL schema. For example, the selection of the agent depends strongly on the observation and action space nature. There are several interfaces to interact with the environments such as OpenAI Gym / Farama Gymnasium or DeepMind. However, each of them has a different API and work with non-compatible data types.

skrl offers a function to **wrap environments** based on the Gym/Gymnasium, DeepMind, Isaac Gym and Omniverse Isaac Gym interfaces (the last two have slight differences with Gym) and offer, for library components, a common interface (based on Gym/Gymnasium) as shown in the following figure. Refer to the :doc:`Wrapping <../modules/skrl.envs.wrapping>` section for more information.

.. image:: ../_static/imgs/wrapping-light.svg
    :width: 100%
    :align: center
    :class: only-light
    :alt: Environment wrapping

.. image:: ../_static/imgs/wrapping-dark.svg
    :width: 100%
    :align: center
    :class: only-dark
    :alt: Environment wrapping

Within the methods and properties defined in the wrapped environment, the observation and action space are one of the most relevant for instantiating other library components. The following code snippets show how to load and wrap environments based on the supported interfaces:

.. tabs::

    .. tab:: Omniverse Isaac Gym

        .. tabs::

            .. tab:: Common environment

                .. code-block:: python

                    # import the environment wrapper and loader
                    from skrl.envs.torch import wrap_env
                    from skrl.envs.torch import load_omniverse_isaacgym_env

                    # load the environment
                    env = load_omniverse_isaacgym_env(task_name="Cartpole")

                    # wrap the environment
                    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="omniverse-isaacgym")'

            .. tab:: Multi-threaded environment

                .. code-block:: python

                    # import the environment wrapper and loader
                    from skrl.envs.torch import wrap_env
                    from skrl.envs.torch import load_omniverse_isaacgym_env

                    # load the multi-threaded environment
                    env = load_omniverse_isaacgym_env(task_name="Cartpole", multi_threaded=True, timeout=30)

                    # wrap the environment
                    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="omniverse-isaacgym")'

    .. tab:: Isaac Gym

        .. tabs::

            .. tab:: Preview 4 (isaacgymenvs.make)

                .. code-block:: python

                    import isaacgymenvs

                    # import the environment wrapper
                    from skrl.envs.torch import wrap_env

                    # create/load the environment using the easy-to-use API from NVIDIA
                    env = isaacgymenvs.make(seed=0,
                                            task="Cartpole",
                                            num_envs=512,
                                            sim_device="cuda:0",
                                            rl_device="cuda:0",
                                            graphics_device_id=0,
                                            headless=False)

                    # wrap the environment
                    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview4")'

            .. tab:: Preview 4

                .. code-block:: python

                    # import the environment wrapper and loader
                    from skrl.envs.torch import wrap_env
                    from skrl.envs.torch import load_isaacgym_env_preview4

                    # load the environment
                    env = load_isaacgym_env_preview4(task_name="Cartpole")

                    # wrap the environment
                    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview4")'

            .. tab:: Preview 3

                .. code-block:: python

                    # import the environment wrapper and loader
                    from skrl.envs.torch import wrap_env
                    from skrl.envs.torch import load_isaacgym_env_preview3

                    # load the environment
                    env = load_isaacgym_env_preview3(task_name="Cartpole")

                    # wrap the environment
                    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview3")'

            .. tab:: Preview 2

                .. code-block:: python

                    # import the environment wrapper and loader
                    from skrl.envs.torch import wrap_env
                    from skrl.envs.torch import load_isaacgym_env_preview2

                    # load the environment
                    env = load_isaacgym_env_preview2(task_name="Cartpole")

                    # wrap the environment
                    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="isaacgym-preview2")'

    .. tab:: Gym / Gymnasium

        .. tabs::

            .. tab:: Gym

                .. tabs::

                    .. tab:: Single environment

                        .. code-block:: python

                            # import the environment wrapper and gym
                            from skrl.envs.torch import wrap_env
                            import gym

                            # load environment
                            env = gym.make('Pendulum-v1')

                            # wrap the environment
                            env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'

                    .. tab:: Vectorized environment

                        Visit the OpenAI Gym documentation (`Vector <https://www.gymlibrary.dev/api/vector>`__) for more information about the creation and usage of vectorized environments.

                        .. code-block:: python

                            # import the environment wrapper and gym
                            from skrl.envs.torch import wrap_env
                            import gym

                            # load a vectorized environment
                            env = gym.vector.make("Pendulum-v1", num_envs=10, asynchronous=False)

                            # wrap the environment
                            env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gym")'

            .. tab:: Gymnasium

                .. tabs::

                    .. tab:: Single environment

                        .. code-block:: python

                            # import the environment wrapper and gymnasium
                            from skrl.envs.torch import wrap_env
                            import gymnasium as gym

                            # load environment
                            env = gym.make('Pendulum-v1')

                            # wrap the environment
                            env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'

                    .. tab:: Vectorized environment

                        Visit the Gymnasium documentation (`Vector <https://gymnasium.farama.org/api/vector>`__) for more information about the creation and usage of vectorized environments.

                        .. code-block:: python

                            # import the environment wrapper and gymnasium
                            from skrl.envs.torch import wrap_env
                            import gymnasium as gym

                            # load a vectorized environment
                            env = gym.vector.make("Pendulum-v1", num_envs=10, asynchronous=False)

                            # wrap the environment
                            env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="gymnasium")'

    .. tab:: DeepMind

        .. code-block:: python

            # import the environment wrapper and the deepmind suite
            from skrl.envs.torch import wrap_env
            from dm_control import suite

            # load environment
            env = suite.load(domain_name="cartpole", task_name="swingup")

            # wrap the environment
            env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="dm")'

Once the environment is known (and instantiated), it is time to configure and instantiate the agent. Agents are composed, apart from the optimization algorithm, by several components, such as memories, models or noises, for example, according to their nature. The following subsections focus on those components.

2. Memories
-----------

Memories are storage components that allow agents to collect and use/reuse recent or past experiences or other types of information. These can be large in size (such as replay buffers used by off-policy algorithms like DDPG, TD3 or SAC) or small in size (such as rollout buffers used by on-policy algorithms like PPO or TRPO to store batches that are discarded after use).

skrl provides **generic memory definitions** that are not tied to the agent implementation and can be used for any role, such as rollout or replay buffers. They are empty shells when they are instantiated and the agents are in charge of defining the tensors according to their needs. The total space occupied is the product of the memory size (:literal:`memory_size`), the number of environments (:literal:`num_envs`) obtained from the wrapped environment and the data size for each defined tensor.

The following code snippets show how to instantiate a memory:

.. tabs::

    .. tab:: Random memory

        .. code-block:: python

            from skrl.memories.torch import RandomMemory

            # instantiate a memory
            memory = RandomMemory(memory_size=100000, num_envs=env.num_envs)

Memories are passed directly to the agent constructor, if required (not all agents require memory, such as Q-learning or SARSA, for example), during its instantiation under the argument :literal:`memory`.

3. Models
---------

Models are the agents' brains. Agents can have one or several models and their parameters are adjusted via the optimization algorithms.

In contrast to other libraries, skrl does not provide predefined models or fixed templates (this practice tends to hide and reduce the flexibility of the system, forcing developers to deeply inspect the code to make even small changes). Nevertheless, **helper classes/mixins are provided** to create discrete and continuous (stochastic or deterministic) models with the library. In this way, the user/researcher should only be concerned with the definition of the approximation functions (tables or artificial neural networks), having all the control in his hands.

The following code snippets show how to define a model, based on the concept of each respective image, using the provided classes/mixins. For more information refer to :ref:`Categorical <models_categorical>`, :ref:`Gaussian <models_gaussian>`, :ref:`Multivariate Gaussian <models_multivariate_gaussian>` and :ref:`Deterministic <models_deterministic>` sections for artificial neural networks models, and :ref:`Tabular <models_tabular>` section for tabular models.

.. tabs::

    .. tab:: Categorical

        .. image:: ../_static/imgs/model_categorical-light.svg
            :width: 100%
            :align: center
            :class: only-light
            :alt: Categorical model

        .. image:: ../_static/imgs/model_categorical-dark.svg
            :width: 100%
            :align: center
            :class: only-dark
            :alt: Categorical model

        .. raw:: html

            <hr>

        .. code-block:: python

            import torch
            import torch.nn as nn
            from skrl.models.torch import Model, CategoricalMixin

            # define the model
            class Policy(CategoricalMixin, Model):
                def __init__(self, observation_space, action_space, device="cuda:0", unnormalized_log_prob=True):
                    Model.__init__(self, observation_space, action_space, device)
                    CategoricalMixin.__init__(self, unnormalized_log_prob)

                    self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                            nn.ELU(),
                                            nn.Linear(32, 32),
                                            nn.ELU(),
                                            nn.Linear(32, self.num_actions))

                def compute(self, inputs, role):
                    return self.net(inputs["states"]), {}

    .. tab:: Gaussian

        .. image:: ../_static/imgs/model_gaussian-light.svg
            :width: 100%
            :align: center
            :class: only-light
            :alt: Gaussian model

        .. image:: ../_static/imgs/model_gaussian-dark.svg
            :width: 100%
            :align: center
            :class: only-dark
            :alt: Gaussian model

        .. raw:: html

            <hr>

        .. code-block:: python

            import torch
            import torch.nn as nn
            from skrl.models.torch import Model, GaussianMixin

            # define the model
            class Policy(GaussianMixin, Model):
                def __init__(self, observation_space, action_space, device="cuda:0",
                             clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
                    Model.__init__(self, observation_space, action_space, device)
                    GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

                    self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                             nn.ELU(),
                                             nn.Linear(32, 32),
                                             nn.ELU(),
                                             nn.Linear(32, self.num_actions))
                    self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

                def compute(self, inputs, role):
                    return self.net(inputs["states"]), self.log_std_parameter, {}

    .. tab:: Multivariate Gaussian

        .. image:: ../_static/imgs/model_multivariate_gaussian-light.svg
            :width: 100%
            :align: center
            :class: only-light
            :alt: Multivariate Gaussian model

        .. image:: ../_static/imgs/model_multivariate_gaussian-dark.svg
            :width: 100%
            :align: center
            :class: only-dark
            :alt: Multivariate Gaussian model

        .. raw:: html

            <hr>

        .. code-block:: python

            import torch
            import torch.nn as nn
            from skrl.models.torch import Model, MultivariateGaussianMixin

            # define the model
            class Policy(MultivariateGaussianMixin, Model):
                def __init__(self, observation_space, action_space, device="cuda:0",
                             clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2):
                    Model.__init__(self, observation_space, action_space, device)
                    MultivariateGaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

                    self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                             nn.ELU(),
                                             nn.Linear(32, 32),
                                             nn.ELU(),
                                             nn.Linear(32, self.num_actions))
                    self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

                def compute(self, inputs, role):
                    return self.net(inputs["states"]), self.log_std_parameter, {}

    .. tab:: Deterministic

        .. image:: ../_static/imgs/model_deterministic-light.svg
            :width: 60%
            :align: center
            :class: only-light
            :alt: Deterministic model

        .. image:: ../_static/imgs/model_deterministic-dark.svg
            :width: 60%
            :align: center
            :class: only-dark
            :alt: Deterministic model

        .. raw:: html

            <hr>

        .. code-block:: python

            import torch
            import torch.nn as nn
            from skrl.models.torch import Model, DeterministicMixin

            # define the model
            class Policy(DeterministicMixin, Model):
                def __init__(self, observation_space, action_space, device="cuda:0", clip_actions=False):
                    Model.__init__(self, observation_space, action_space, device)
                    DeterministicMixin.__init__(self, clip_actions)

                    self.net = nn.Sequential(nn.Linear(self.num_observations, 32),
                                             nn.ELU(),
                                             nn.Linear(32, 32),
                                             nn.ELU(),
                                             nn.Linear(32, self.num_actions))

                def compute(self, inputs, role):
                    return self.net(inputs["states"]), {}

    .. tab:: Tabular

        .. code-block:: python

            import torch
            from skrl.models.torch import Model, TabularMixin

            # define the model
            class Policy(TabularMixin, Model):
                def __init__(self, observation_space, action_space, device="cuda:0", num_envs=1):
                    Model.__init__(self, observation_space, action_space, device)
                    TabularMixin.__init__(self, num_envs)

                    self.table = torch.ones((num_envs, self.num_observations, self.num_actions),
                                            dtype=torch.float32, device=self.device)

                def compute(self, inputs, role):
                    actions = torch.argmax(self.table[torch.arange(self.num_envs).view(-1, 1), inputs["states"]],
                                           dim=-1, keepdim=True).view(-1,1)
                    return actions, {}

Models must be collected in a dictionary and passed to the agent constructor during its instantiation under the argument :literal:`models`. The dictionary keys are specific to each agent. Visit their respective documentation for more details (under *Spaces and models* section). For example, the PPO agent requires the policy and value models as shown below:

.. code-block:: python

    models = {}
    models["policy"] = Policy(env.observation_space, env.action_space, env.device)
    models["value"] = Value(env.observation_space, env.action_space, env.device)

Models can be saved and loaded to and from the file system. However, the recommended practice for loading checkpoints to perform evaluations or continue an interrupted training is through the agents (they include, in addition to the models, other components and internal instances such as preprocessors or optimizers). Refer to :ref:`Saving, loading and logging <data>` (under *Checkpoints* section) for more information.

4. Noises
---------

Noise plays a fundamental role in the exploration stage, especially in agents of a deterministic nature, such as DDPG or TD3, for example.

skrl provides, as part of its resources, **classes for instantiating noises** as shown in the following code snippets. Refer to :ref:`Noises <resources_noises>` documentation for more information.

.. tabs::

    .. tab:: Gaussian noise

        .. code-block:: python

            from skrl.resources.noises.torch import GaussianNoise

            # instantiate a noise
            noise = GaussianNoise(mean=0, std=0.2, device=env.device)

    .. tab:: Ornstein-Uhlenbeck noise

        .. code-block:: python

            from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise

            # instantiate a noise
            noise = OrnsteinUhlenbeckNoise(theta=0.15, sigma=0.2, base_scale=1.0, device=env.device)

Noise instances are passed to the agents in their respective configuration dictionaries. For example, the DDPG agent requires the exploration noise as shown below:

.. code-block:: python

    from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG

    agent_cfg = DDPG_DEFAULT_CONFIG.copy()
    agent_cfg["exploration"]["noise"] = noise

5. Learning rate schedulers
---------------------------

Learning rate schedulers help RL system converge faster and improve accuracy.

skrl **supports all PyTorch learning rate schedulers** and provides, as part of its resources, **additional schedulers**. Refer to :ref:`Learning rate schedulers <resources_schedulers>` documentation for more information.

Learning rate schedulers classes and their respective arguments (except the :literal:`optimizer` argument) are passed to the agents in their respective configuration dictionaries. For example, for the PPO agent, one of the schedulers can be configured as shown below:

.. code-block:: python

    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.resources.schedulers.torch import KLAdaptiveRL

    agent_cfg = PPO_DEFAULT_CONFIG.copy()
    agent_cfg["learning_rate_scheduler"] = KLAdaptiveRL
    agent_cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}

6. Preprocessors
----------------

Data preprocessing can help increase the accuracy and efficiency of training by cleaning or making data suitable for machine learning models.

skrl provides, as part of its resources, **preprocessors** classes. Refer to :ref:`Preprocessors <resources_preprocessors>` documentation for more information.

Preprocessors classes and their respective arguments are passed to the agents in their respective configuration dictionaries. For example, for the PPO agent, one of the preprocessors can be configured as shown below:

.. code-block:: python

    from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
    from skrl.resources.preprocessors.torch import RunningStandardScaler

    agent_cfg["state_preprocessor"] = RunningStandardScaler
    agent_cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": env.device}
    agent_cfg["value_preprocessor"] = RunningStandardScaler
    agent_cfg["value_preprocessor_kwargs"] = {"size": 1, "device": env.device}

7. Agents
---------

Agents are the components in charge of decision making. They are much more than models (neural networks, for example) and include the optimization algorithms that compute the optimal policy

skrl provides **state-of-the-art agent**. Its implementations are focused on readability, simplicity and code transparency. Each agent is implemented independently even when two or more agents may contain code in common. Refer to each agent documentation for more information about the models and spaces they support, their respective configurations, algorithm details and more.

    * :doc:`Advantage Actor Critic <../modules/skrl.agents.a2c>` (**A2C**)
    * :doc:`Adversarial Motion Priors <../modules/skrl.agents.amp>` (**AMP**)
    * :doc:`Cross-Entropy Method <../modules/skrl.agents.cem>` (**CEM**)
    * :doc:`Deep Deterministic Policy Gradient <../modules/skrl.agents.ddpg>` (**DDPG**)
    * :doc:`Double Deep Q-Network <../modules/skrl.agents.ddqn>` (**DDQN**)
    * :doc:`Deep Q-Network <../modules/skrl.agents.dqn>` (**DQN**)
    * :doc:`Proximal Policy Optimization <../modules/skrl.agents.ppo>` (**PPO**)
    * :doc:`Q-learning <../modules/skrl.agents.q_learning>` (**Q-learning**)
    * :doc:`Soft Actor-Critic <../modules/skrl.agents.sac>` (**SAC**)
    * :doc:`State Action Reward State Action <../modules/skrl.agents.sarsa>` (**SARSA**)
    * :doc:`Twin-Delayed DDPG <../modules/skrl.agents.td3>` (**TD3**)
    * :doc:`Trust Region Policy Optimization <../modules/skrl.agents.trpo>` (**TRPO**)

Agents generally expect, as arguments, the following components: models and memories, as well as the following variables: observation and action spaces, the device where their logic is executed and a configuration dictionary with hyperparameters and other values. The remaining components, mentioned above, are collected through the configuration dictionary. For example, the PPO agent can be instantiated as follows:

.. code-block:: python

    from skrl.agents.torch.ppo import PPO

    agent = PPO(models=models,  # models dict
                memory=memory,  # memory instance, or None if not required
                cfg=agent_cfg,  # configuration dict (preprocessors, learning rate schedulers, etc.)
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)

Agents can be saved and loaded to and from the file system. This is the **recommended practice** for loading checkpoints to perform evaluations or to continue interrupted training (since they include, in addition to models, other internal components and instances such as preprocessors or optimizers). Refer to :ref:`Saving, loading and logging <data>` (under *Checkpoints* section) for more information.

8. Trainers
-----------

Now that both actors, the environment and the agent, are instantiated, it is time to put the RL system in motion.

skrl offers classes (called **trainers**) that manage the interaction cycle between the environment and the agent(s) for both: training and evaluation. These classes also enable the simultaneous training and evaluation of several agents by scope (subsets of environments among all available environments), which may or may not share resources, in the same run.

The following code snippets show how to load and wrap environments based on the supported interfaces:

.. tabs::

    .. tab:: Sequential trainer

        .. code-block:: python

            from skrl.trainers.torch import SequentialTrainer

            # create a sequential trainer
            cfg = {"timesteps": 50000, "headless": False}
            trainer = SequentialTrainer(env=env, agents=[agent], cfg=cfg)

            # train the agent(s)
            trainer.train()

            # evaluate the agent(s)
            trainer.eval()

    .. tab:: Parallel trainer

        .. code-block:: python

            from skrl.trainers.torch import ParallelTrainer

            # create a parallel trainer
            cfg = {"timesteps": 50000, "headless": False}
            trainer = ParallelTrainer(env=env, agents=[agent], cfg=cfg)

            # train the agent(s)
            trainer.train()

            # evaluate the agent(s)
            trainer.eval()

    .. tab:: Manual trainer

        .. code-block:: python

            from skrl.trainers.torch import ManualTrainer

            # create a manual trainer
            cfg = {"timesteps": 50000, "headless": False}
            trainer = ManualTrainer(env=env, agents=[agent], cfg=cfg)

            # train the agent(s)
            trainer.train()

            # evaluate the agent(s)
            trainer.eval()

.. raw:: html

    <hr>

**What's next?**

Visit the :ref:`Examples <examples>` section for training and evaluation demonstrations with different environment interfaces and highlighted practices, among others.
