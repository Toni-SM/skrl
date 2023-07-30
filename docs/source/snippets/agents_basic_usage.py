# [torch-start-a2c]
# import the agent and its default configuration
from skrl.agents.torch.a2c import A2C, A2C_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = A2C_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = A2C(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [torch-end-a2c]


# [jax-start-a2c]
# import the agent and its default configuration
from skrl.agents.jax.a2c import A2C, A2C_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = A2C_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = A2C(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [jax-end-a2c]


# [torch-start-a2c-rnn]
# import the agent and its default configuration
from skrl.agents.torch.a2c import A2C_RNN as A2C, A2C_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = A2C_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = A2C(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [torch-end-a2c-rnn]

# =============================================================================

# [torch-start-amp]
# import the agent and its default configuration
from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training
models["discriminator"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = AMP_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
# (assuming defined memories for motion <motion_dataset> and <reply_buffer>)
# (assuming defined methods to collect motion <collect_reference_motions> and <collect_observation>)
agent = AMP(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            amp_observation_space=env.amp_observation_space,
            motion_dataset=motion_dataset,
            reply_buffer=reply_buffer,
            collect_reference_motions=collect_reference_motions,
            collect_observation=collect_observation)
# [torch-end-amp]

# =============================================================================

# [torch-start-cem]
# import the agent and its default configuration
from skrl.agents.torch.cem import CEM, CEM_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...

# adjust some configuration if necessary
cfg_agent = CEM_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = CEM(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [torch-end-cem]


# [jax-start-cem]
# import the agent and its default configuration
from skrl.agents.jax.cem import CEM, CEM_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...

# adjust some configuration if necessary
cfg_agent = CEM_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = CEM(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [jax-end-cem]

# =============================================================================

# [torch-start-ddpg]
# import the agent and its default configuration
from skrl.agents.torch.ddpg import DDPG, DDPG_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["target_policy"] = ...  # only required during training
models["critic"] = ...  # only required during training
models["target_critic"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = DDPG_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = DDPG(models=models,
             memory=memory,  # only required during training
             cfg=cfg_agent,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device)
# [torch-end-ddpg]


# [jax-start-ddpg]
# import the agent and its default configuration
from skrl.agents.jax.ddpg import DDPG, DDPG_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["target_policy"] = ...  # only required during training
models["critic"] = ...  # only required during training
models["target_critic"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = DDPG_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = DDPG(models=models,
             memory=memory,  # only required during training
             cfg=cfg_agent,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device)
# [jax-end-ddpg]


# [torch-start-ddpg-rnn]
# import the agent and its default configuration
from skrl.agents.torch.ddpg import DDPG_RNN as DDPG, DDPG_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["target_policy"] = ...  # only required during training
models["critic"] = ...  # only required during training
models["target_critic"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = DDPG_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = DDPG(models=models,
             memory=memory,  # only required during training
             cfg=cfg_agent,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device)
# [torch-end-ddpg-rnn]

# =============================================================================

# [torch-start-ddqn]
# import the agent and its default configuration
from skrl.agents.torch.dqn import DDQN, DDQN_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["q_network"] = ...
models["target_q_network"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = DDQN_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = DDQN(models=models,
             memory=memory,  # only required during training
             cfg=cfg_agent,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device)
# [torch-end-ddqn]


# [jax-start-ddqn]
# import the agent and its default configuration
from skrl.agents.jax.dqn import DDQN, DDQN_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["q_network"] = ...
models["target_q_network"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = DDQN_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = DDQN(models=models,
             memory=memory,  # only required during training
             cfg=cfg_agent,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device)
# [jax-end-ddqn]

# =============================================================================

# [torch-start-dqn]
# import the agent and its default configuration
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["q_network"] = ...
models["target_q_network"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = DQN_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = DQN(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [torch-end-dqn]


# [jax-start-dqn]
# import the agent and its default configuration
from skrl.agents.jax.dqn import DQN, DQN_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["q_network"] = ...
models["target_q_network"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = DQN_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = DQN(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [jax-end-dqn]

# =============================================================================

# [torch-start-ppo]
# import the agent and its default configuration
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = PPO_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = PPO(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [torch-end-ppo]


# [jax-start-ppo]
# import the agent and its default configuration
from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = PPO_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = PPO(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [jax-end-ppo]


# [torch-start-ppo-rnn]
# import the agent and its default configuration
from skrl.agents.torch.ppo import PPO_RNN as PPO, PPO_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = PPO_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = PPO(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [torch-end-ppo-rnn]

# =============================================================================

# [torch-start-q-learning]
# import the agent and its default configuration
from skrl.agents.torch.q_learning import Q_LEARNING, Q_LEARNING_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...

# adjust some configuration if necessary
cfg_agent = Q_LEARNING_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env>)
agent = Q_LEARNING(models=models,
                   memory=None,
                   cfg=cfg_agent,
                   observation_space=env.observation_space,
                   action_space=env.action_space,
                   device=env.device)
# [torch-end-q-learning]

# =============================================================================

# [torch-start-rpo-with-rpo]
class Policy(GaussianMixin, Model):
    ...

    def compute(self, inputs, role):
        # compute the mean actions using the neural network
        mean_actions = self.net(inputs["states"])

        # perturb the mean actions by adding a randomized uniform sample
        rpo_alpha = inputs["alpha"]
        perturbation = torch.zeros_like(mean_actions).uniform_(-rpo_alpha, rpo_alpha)
        mean_actions += perturbation

        return mean_actions, self.log_std_parameter, {}
# [torch-end-rpo-with-rpo]


# [jax-start-rpo-with-rpo]
class Policy(GaussianMixin, Model):
    ...

    def __call__(self, inputs, role):
        # compute the mean actions using the neural network
        mean_actions = ...
        log_std = ...

        # perturb the mean actions by adding a randomized uniform sample
        rpo_alpha = inputs["alpha"]
        perturbation = jax.random.uniform(inputs["key"], mean_actions.shape, minval=-rpo_alpha, maxval=rpo_alpha)
        mean_actions += perturbation

        return mean_actions, log_std, {}
# [jax-end-rpo-with-rpo]


# [torch-start-rpo-without-rpo]
class Policy(GaussianMixin, Model):
    ...

    def compute(self, inputs, role):
        # compute the mean actions using the neural network
        mean_actions = self.net(inputs["states"])

        # perturb the mean actions by adding a randomized uniform sample
        rpo_alpha = 0.5
        perturbation = torch.zeros_like(mean_actions).uniform_(-rpo_alpha, rpo_alpha)
        mean_actions += perturbation

        return mean_actions, self.log_std_parameter, {}
# [torch-end-rpo-without-rpo]


# [jax-start-rpo-without-rpo]
class Policy(GaussianMixin, Model):
    ...

    def __call__(self, inputs, role):
        # compute the mean actions using the neural network
        mean_actions = ...
        log_std = ...

        # perturb the mean actions by adding a randomized uniform sample
        rpo_alpha = 0.5
        perturbation = jax.random.uniform(inputs["key"], mean_actions.shape, minval=-rpo_alpha, maxval=rpo_alpha)
        mean_actions += perturbation

        return mean_actions, log_std, {}
# [jax-end-rpo-without-rpo]


# [torch-start-rpo]
# import the agent and its default configuration
from skrl.agents.torch.rpo import RPO, RPO_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = RPO_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = RPO(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [torch-end-rpo]


# [jax-start-rpo]
# import the agent and its default configuration
from skrl.agents.jax.rpo import RPO, RPO_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = RPO_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = RPO(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [jax-end-rpo]


# [torch-start-rpo-rnn]
# import the agent and its default configuration
from skrl.agents.torch.rpo import RPO_RNN as RPO, RPO_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = RPO_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = RPO(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [torch-end-rpo-rnn]

# =============================================================================

# [torch-start-sac]
# import the agent and its default configuration
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["critic_1"] = ...  # only required during training
models["critic_2"] = ...  # only required during training
models["target_critic_1"] = ...  # only required during training
models["target_critic_2"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = SAC_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = SAC(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [torch-end-sac]


# [jax-start-sac]
# import the agent and its default configuration
from skrl.agents.jax.sac import SAC, SAC_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["critic_1"] = ...  # only required during training
models["critic_2"] = ...  # only required during training
models["target_critic_1"] = ...  # only required during training
models["target_critic_2"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = SAC_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = SAC(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [jax-end-sac]


# [torch-start-sac-rnn]
# import the agent and its default configuration
from skrl.agents.torch.sac import SAC_RNN as SAC, SAC_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["critic_1"] = ...  # only required during training
models["critic_2"] = ...  # only required during training
models["target_critic_1"] = ...  # only required during training
models["target_critic_2"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = SAC_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = SAC(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [torch-end-sac-rnn]

# =============================================================================

# [torch-start-sarsa]
# import the agent and its default configuration
from skrl.agents.torch.sarsa import SARSA, SARSA_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...

# adjust some configuration if necessary
cfg_agent = SARSA_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env>)
agent = SARSA(models=models,
              memory=None,
              cfg=cfg_agent,
              observation_space=env.observation_space,
              action_space=env.action_space,
              device=env.device)
# [torch-end-sarsa]

# =============================================================================

# [torch-start-td3]
# import the agent and its default configuration
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["target_policy"] = ...  # only required during training
models["critic_1"] = ...  # only required during training
models["critic_2"] = ...  # only required during training
models["target_critic_1"] = ...  # only required during training
models["target_critic_2"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = TD3_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = TD3(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [torch-end-td3]


# [jax-start-td3]
# import the agent and its default configuration
from skrl.agents.jax.td3 import TD3, TD3_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["target_policy"] = ...  # only required during training
models["critic_1"] = ...  # only required during training
models["critic_2"] = ...  # only required during training
models["target_critic_1"] = ...  # only required during training
models["target_critic_2"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = TD3_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = TD3(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [jax-end-td3]


# [torch-start-td3-rnn]
# import the agent and its default configuration
from skrl.agents.torch.td3 import TD3_RNN as TD3, TD3_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["target_policy"] = ...  # only required during training
models["critic_1"] = ...  # only required during training
models["critic_2"] = ...  # only required during training
models["target_critic_1"] = ...  # only required during training
models["target_critic_2"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = TD3_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = TD3(models=models,
            memory=memory,  # only required during training
            cfg=cfg_agent,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
# [torch-end-td3-rnn]

# =============================================================================

# [torch-start-trpo]
# import the agent and its default configuration
from skrl.agents.torch.trpo import TRPO, TRPO_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = TRPO_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = TRPO(models=models,
             memory=memory,  # only required during training
             cfg=cfg_agent,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device)
# [torch-end-trpo]


# [torch-start-trpo-rnn]
# import the agent and its default configuration
from skrl.agents.torch.trpo import TRPO_RNN as TRPO, TRPO_DEFAULT_CONFIG

# instantiate the agent's models
models = {}
models["policy"] = ...
models["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = TRPO_DEFAULT_CONFIG.copy()
cfg_agent["<KEY>"] = ...

# instantiate the agent
# (assuming a defined environment <env> and memory <memory>)
agent = TRPO(models=models,
             memory=memory,  # only required during training
             cfg=cfg_agent,
             observation_space=env.observation_space,
             action_space=env.action_space,
             device=env.device)
# [torch-end-trpo-rnn]
