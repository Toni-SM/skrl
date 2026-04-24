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
