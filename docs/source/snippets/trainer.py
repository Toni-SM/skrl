# [pytorch-start-sequential]
from skrl.trainers.torch import SequentialTrainer

# assuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = SequentialTrainer(env=env, agents=agents, cfg=cfg)

# train the agent(s)
trainer.train()

# evaluate the agent(s)
trainer.eval()
# [pytorch-end-sequential]


# [jax-start-sequential]
from skrl.trainers.jax import SequentialTrainer

# assuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = SequentialTrainer(env=env, agents=agents, cfg=cfg)

# train the agent(s)
trainer.train()

# evaluate the agent(s)
trainer.eval()
# [jax-end-sequential]


# [warp-start-sequential]
from skrl.trainers.warp import SequentialTrainer

# assuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = SequentialTrainer(env=env, agents=agents, cfg=cfg)

# train the agent(s)
trainer.train()

# evaluate the agent(s)
trainer.eval()
# [warp-end-sequential]

# =============================================================================

# [pytorch-start-parallel]
from skrl.trainers.torch import ParallelTrainer

# assuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = ParallelTrainer(env=env, agents=agents, cfg=cfg)

# train the agent(s)
trainer.train()

# evaluate the agent(s)
trainer.eval()
# [pytorch-end-parallel]

# =============================================================================

# [pytorch-start-step]
from skrl.trainers.torch import StepTrainer

# assuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = StepTrainer(env=env, agents=agents, cfg=cfg)

# train the agent(s)
for timestep in range(cfg["timesteps"]):
    trainer.train(timestep=timestep)

# evaluate the agent(s)
for timestep in range(cfg["timesteps"]):
    trainer.eval(timestep=timestep)
# [pytorch-end-step]


# [jax-start-step]
from skrl.trainers.jax import StepTrainer

# assuming there is an environment called 'env'
# and an agent or a list of agents called 'agents'

# create a sequential trainer
cfg = {"timesteps": 50000, "headless": False}
trainer = StepTrainer(env=env, agents=agents, cfg=cfg)

# train the agent(s)
for timestep in range(cfg["timesteps"]):
    trainer.train(timestep=timestep)

# evaluate the agent(s)
for timestep in range(cfg["timesteps"]):
    trainer.eval(timestep=timestep)
# [jax-end-step]

# =============================================================================

# [pytorch-start-manual-training]

# [pytorch-end-manual-training]

# [pytorch-start-manual-evaluation]
# assuming there is an environment named 'env'
# and an agent named 'agents' (or a state-preprocessor and a policy)

states, infos = env.reset()

for i in range(1000):
    # state-preprocessor + policy
    with torch.no_grad():
        states = state_preprocessor(states)
        actions = policy.act({"states": states})[0]

    # step the environment
    next_states, rewards, terminated, truncated, infos = env.step(actions)

    # render the environment
    env.render()

    # check for termination/truncation
    if terminated.any() or truncated.any():
        states, infos = env.reset()
    else:
        states = next_states
# [pytorch-end-manual-evaluation]


# [jax-start-manual-training]

# [jax-end-manual-training]

# [jax-start-manual-evaluation]

# [jax-end-manual-evaluation]
