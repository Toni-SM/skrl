# [start-ippo-torch]
# import the agent and its default configuration
from skrl.multi_agents.torch.ippo import IPPO, IPPO_CFG

# instantiate the agent's models
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = IPPO_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = IPPO(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-ippo-torch]


# [start-ippo-jax]
# import the agent and its default configuration
from skrl.multi_agents.jax.ippo import IPPO, IPPO_CFG

# instantiate the agent's models
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = IPPO_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = IPPO(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-ippo-jax]


# [start-mappo-torch]
# import the agent and its default configuration
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG

# instantiate the agent's models
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = MAPPO_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = MAPPO(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-mappo-torch]


# [start-mappo-jax]
# import the agent and its default configuration
from skrl.multi_agents.jax.mappo import MAPPO, MAPPO_CFG

# instantiate the agent's models
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["value"] = ...  # only required during training

# adjust some configuration if necessary
cfg_agent = MAPPO_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = MAPPO(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-mappo-jax]


# [start-happo-torch]
# import the agent and its default configuration
from skrl.multi_agents.torch.happo import HAPPO, HAPPO_CFG

# instantiate the agent's models
models = {}
value = ...  # value model shared by all agents (alternatively, a distinct value model can be given to each agent)
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["value"] = value  # only required during training

# adjust some configuration if necessary
cfg_agent = HAPPO_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = HAPPO(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-happo-torch]


# [start-haa2c-torch]
# import the agent and its default configuration
from skrl.multi_agents.torch.haa2c import HAA2C, HAA2C_CFG

# instantiate the agent's models
models = {}
value = ...  # value model shared by all agents (alternatively, a distinct value model can be given to each agent)
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["value"] = value  # only required during training

# adjust some configuration if necessary
cfg_agent = HAA2C_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = HAA2C(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-haa2c-torch]


# [start-hatrpo-torch]
# import the agent and its default configuration
from skrl.multi_agents.torch.hatrpo import HATRPO, HATRPO_CFG

# instantiate the agent's models
models = {}
value = ...  # value model shared by all agents (alternatively, a distinct value model can be given to each agent)
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["value"] = value  # only required during training

# adjust some configuration if necessary
cfg_agent = HATRPO_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = HATRPO(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-hatrpo-torch]


# [start-haddpg-torch]
# import the agent and its default configuration
from skrl.multi_agents.torch.haddpg import HADDPG, HADDPG_CFG
from skrl.utils.spaces.torch import compute_joint_space

# instantiate the agent's models
# - the critic takes the joint action of all agents, so it must be built with the joint action space
joint_action_space = compute_joint_space(
    [env.action_space(agent_name) for agent_name in env.possible_agents], mode="concatenate"
)
critic = ...  # critic shared by all agents (alternatively, a distinct critic can be given to each agent)
target_critic = ...  # target critic shared by all agents (alternatively, one per agent)
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["target_policy"] = ...  # only required during training
    models[agent_name]["critic"] = critic  # only required during training
    models[agent_name]["target_critic"] = target_critic  # only required during training

# adjust some configuration if necessary
cfg_agent = HADDPG_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = HADDPG(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-haddpg-torch]


# [start-hatd3-torch]
# import the agent and its default configuration
from skrl.multi_agents.torch.hatd3 import HATD3, HATD3_CFG
from skrl.utils.spaces.torch import compute_joint_space

# instantiate the agent's models
# - the critics take the joint action of all agents, so they must be built with the joint action space
joint_action_space = compute_joint_space(
    [env.action_space(agent_name) for agent_name in env.possible_agents], mode="concatenate"
)
critic_1 = ...  # critics shared by all agents (alternatively, distinct critics can be given to each agent)
critic_2 = ...
target_critic_1 = ...
target_critic_2 = ...
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["target_policy"] = ...  # only required during training
    models[agent_name]["critic_1"] = critic_1  # only required during training
    models[agent_name]["critic_2"] = critic_2  # only required during training
    models[agent_name]["target_critic_1"] = target_critic_1  # only required during training
    models[agent_name]["target_critic_2"] = target_critic_2  # only required during training

# adjust some configuration if necessary
cfg_agent = HATD3_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = HATD3(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-hatd3-torch]


# [start-hasac-torch]
# import the agent and its default configuration
from skrl.multi_agents.torch.hasac import HASAC, HASAC_CFG
from skrl.utils.spaces.torch import compute_joint_space

# instantiate the agent's models
# - the critics take the joint action of all agents, so they must be built with the joint action space
joint_action_space = compute_joint_space(
    [env.action_space(agent_name) for agent_name in env.possible_agents], mode="concatenate"
)
critic_1 = ...  # critics shared by all agents (alternatively, distinct critics can be given to each agent)
critic_2 = ...
target_critic_1 = ...
target_critic_2 = ...
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["critic_1"] = critic_1  # only required during training
    models[agent_name]["critic_2"] = critic_2  # only required during training
    models[agent_name]["target_critic_1"] = target_critic_1  # only required during training
    models[agent_name]["target_critic_2"] = target_critic_2  # only required during training

# adjust some configuration if necessary
cfg_agent = HASAC_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = HASAC(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-hasac-torch]


# [start-maddpg-torch]
# import the agent and its default configuration
from skrl.multi_agents.torch.maddpg import MADDPG, MADDPG_CFG
from skrl.utils.spaces.torch import compute_joint_space

# instantiate the agent's models
# - the critic takes the joint action of all agents, so it must be built with the joint action space
joint_action_space = compute_joint_space(
    [env.action_space(agent_name) for agent_name in env.possible_agents], mode="concatenate"
)
critic = ...  # critic shared by all agents (alternatively, a distinct critic can be given to each agent)
target_critic = ...  # target critic shared by all agents (alternatively, one per agent)
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["target_policy"] = ...  # only required during training
    models[agent_name]["critic"] = critic  # only required during training
    models[agent_name]["target_critic"] = target_critic  # only required during training

# adjust some configuration if necessary
cfg_agent = MADDPG_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = MADDPG(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-maddpg-torch]


# [start-matd3-torch]
# import the agent and its default configuration
from skrl.multi_agents.torch.matd3 import MATD3, MATD3_CFG
from skrl.utils.spaces.torch import compute_joint_space

# instantiate the agent's models
# - the critics take the joint action of all agents, so they must be built with the joint action space
joint_action_space = compute_joint_space(
    [env.action_space(agent_name) for agent_name in env.possible_agents], mode="concatenate"
)
critic_1 = ...  # critics shared by all agents (alternatively, distinct critics can be given to each agent)
critic_2 = ...
target_critic_1 = ...
target_critic_2 = ...
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = ...
    models[agent_name]["target_policy"] = ...  # only required during training
    models[agent_name]["critic_1"] = critic_1  # only required during training
    models[agent_name]["critic_2"] = critic_2  # only required during training
    models[agent_name]["target_critic_1"] = target_critic_1  # only required during training
    models[agent_name]["target_critic_2"] = target_critic_2  # only required during training

# adjust some configuration if necessary
cfg_agent = MATD3_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = MATD3(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-matd3-torch]


# [start-had3qn-torch]
# import the agent and its default configuration
from skrl.multi_agents.torch.had3qn import HAD3QN, HAD3QN_CFG
from skrl.utils.spaces.torch import compute_joint_space

# instantiate the agent's models
# - the critic outputs a value for each joint action of all agents (the product of the agents' numbers of actions),
#   so it must be built with the joint action space
joint_action_space = compute_joint_space(
    [env.action_space(agent_name) for agent_name in env.possible_agents], mode="product"
)
critic = ...  # critic shared by all agents (required)
target_critic = ...  # target critic shared by all agents (required)
models = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["q_network"] = ...
    models[agent_name]["target_q_network"] = ...  # only required during training
    models[agent_name]["critic"] = critic  # only required during training
    models[agent_name]["target_critic"] = target_critic  # only required during training

# adjust some configuration if necessary
cfg_agent = HAD3QN_CFG()
cfg_agent.KEY = ...

# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = HAD3QN(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,  # only required during training
    cfg=cfg_agent,
    observation_spaces=env.observation_spaces,
    state_spaces=env.state_spaces,
    action_spaces=env.action_spaces,
    device=env.device,
)
# [end-had3qn-torch]
