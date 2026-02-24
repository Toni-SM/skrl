# [start-random-torch]
# import the memory class
from skrl.memories.torch import RandomMemory

# instantiate the memory (given a wrapped environment: `env`)
# for 1000 samples per environment for example (note that, some algorithms
# require the capacity to be equal to the number of rollouts, such as PPO)
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=env.device)
# [end-random-torch]


# [start-random-jax]
# import the memory class
from skrl.memories.jax import RandomMemory

# instantiate the memory (given a wrapped environment: `env`)
# for 1000 samples per environment for example (note that, some algorithms
# require the capacity to be equal to the number of rollouts, such as PPO)
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=env.device)
# [end-random-jax]


# [start-random-warp]
# import the memory class
from skrl.memories.warp import RandomMemory

# instantiate the memory (given a wrapped environment: `env`)
# for 1000 samples per environment for example (note that, some algorithms
# require the capacity to be equal to the number of rollouts, such as PPO)
memory = RandomMemory(memory_size=1000, num_envs=env.num_envs, device=env.device)
# [end-random-warp]
