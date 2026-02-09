# [start-random-torch]
# import the memory class
from skrl.memories.torch import RandomMemory

# instantiate the memory (assumes there is a wrapped environment: env)
# - define capacity for 1000 samples per environment (some algorithms, like PPO,
#   require the capacity to be equal to the number of rollouts)
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=env.device)
# [end-random-torch]


# [start-random-jax]
# import the memory class
from skrl.memories.jax import RandomMemory

# instantiate the memory (assumes there is a wrapped environment: env)
# - define capacity for 1000 samples per environment (some algorithms, like PPO,
#   require the capacity to be equal to the number of rollouts)
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=env.device)
# [end-random-jax]


# [start-random-warp]
# import the memory class
from skrl.memories.warp import RandomMemory

# instantiate the memory (assumes there is a wrapped environment: env)
# - define capacity for 1000 samples per environment (some algorithms, like PPO,
#   require the capacity to be equal to the number of rollouts)
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=env.device)
# [end-random-warp]
