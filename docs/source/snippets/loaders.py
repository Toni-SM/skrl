# [start-omniverse-isaac-gym-envs-parameters-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env

# load environment
env = load_omniverse_isaacgym_env(task_name="Cartpole")
# [end-omniverse-isaac-gym-envs-parameters-torch]


# [start-omniverse-isaac-gym-envs-parameters-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_omniverse_isaacgym_env

# load environment
env = load_omniverse_isaacgym_env(task_name="Cartpole")
# [end-omniverse-isaac-gym-envs-parameters-jax]


# [start-omniverse-isaac-gym-envs-cli-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env

# load environment
env = load_omniverse_isaacgym_env()
# [end-omniverse-isaac-gym-envs-cli-torch]


# [start-omniverse-isaac-gym-envs-cli-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_omniverse_isaacgym_env

# load environment
env = load_omniverse_isaacgym_env()
# [end-omniverse-isaac-gym-envs-cli-jax]


# [start-omniverse-isaac-gym-envs-multi-threaded-parameters-torch]
import threading

# import the environment loader
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env

# load environment
env = load_omniverse_isaacgym_env(task_name="Cartpole", multi_threaded=True, timeout=30)

# ...

# start training in a separate thread
threading.Thread(target=trainer.train).start()

# run the simulation in the main thread
env.run()
# [end-omniverse-isaac-gym-envs-multi-threaded-parameters-torch]


# [start-omniverse-isaac-gym-envs-multi-threaded-parameters-jax]
import threading

# import the environment loader
from skrl.envs.loaders.jax import load_omniverse_isaacgym_env

# load environment
env = load_omniverse_isaacgym_env(task_name="Cartpole", multi_threaded=True, timeout=30)

# ...

# start training in a separate thread
threading.Thread(target=trainer.train).start()

# run the simulation in the main thread
env.run()
# [end-omniverse-isaac-gym-envs-multi-threaded-parameters-jax]


# [start-omniverse-isaac-gym-envs-multi-threaded-cli-torch]
import threading

# import the environment loader
from skrl.envs.loaders.torch import load_omniverse_isaacgym_env

# load environment
env = load_omniverse_isaacgym_env(multi_threaded=True, timeout=30)

# ...

# start training in a separate thread
threading.Thread(target=trainer.train).start()

# run the simulation in the main thread
env.run()
# [end-omniverse-isaac-gym-envs-multi-threaded-cli-torch]


# [start-omniverse-isaac-gym-envs-multi-threaded-cli-jax]
import threading

# import the environment loader
from skrl.envs.loaders.jax import load_omniverse_isaacgym_env

# load environment
env = load_omniverse_isaacgym_env(multi_threaded=True, timeout=30)

# ...

# start training in a separate thread
threading.Thread(target=trainer.train).start()

# run the simulation in the main thread
env.run()
# [end-omniverse-isaac-gym-envs-multi-threaded-cli-jax]

# =============================================================================

# [start-isaac-orbit-envs-parameters-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_isaac_orbit_env

# load environment
env = load_isaac_orbit_env(task_name="Isaac-Cartpole-v0")
# [end-isaac-orbit-envs-parameters-torch]


# [start-isaac-orbit-envs-parameters-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_isaac_orbit_env

# load environment
env = load_isaac_orbit_env(task_name="Isaac-Cartpole-v0")
# [end-isaac-orbit-envs-parameters-jax]


# [start-isaac-orbit-envs-cli-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_isaac_orbit_env

# load environment
env = load_isaac_orbit_env()
# [end-isaac-orbit-envs-cli-torch]


# [start-isaac-orbit-envs-cli-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_isaac_orbit_env

# load environment
env = load_isaac_orbit_env()
# [end-isaac-orbit-envs-cli-jax]

# =============================================================================

# [start-isaac-gym-envs-preview-4-api]
import isaacgymenvs

env = isaacgymenvs.make(seed=0,
                        task="Cartpole",
                        num_envs=2000,
                        sim_device="cuda:0",
                        rl_device="cuda:0",
                        graphics_device_id=0,
                        headless=False)
# [end-isaac-gym-envs-preview-4-api]


# [start-isaac-gym-envs-preview-4-parameters-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_isaacgym_env_preview4

# load environment
env = load_isaacgym_env_preview4(task_name="Cartpole")
# [end-isaac-gym-envs-preview-4-parameters-torch]


# [start-isaac-gym-envs-preview-4-parameters-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_isaacgym_env_preview4

# load environment
env = load_isaacgym_env_preview4(task_name="Cartpole")
# [end-isaac-gym-envs-preview-4-parameters-jax]


# [start-isaac-gym-envs-preview-4-cli-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_isaacgym_env_preview4

# load environment
env = load_isaacgym_env_preview4()
# [end-isaac-gym-envs-preview-4-cli-torch]


# [start-isaac-gym-envs-preview-4-cli-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_isaacgym_env_preview4

# load environment
env = load_isaacgym_env_preview4()
# [end-isaac-gym-envs-preview-4-cli-jax]


# [start-isaac-gym-envs-preview-3-parameters-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_isaacgym_env_preview3

# load environment
env = load_isaacgym_env_preview3(task_name="Cartpole")
# [end-isaac-gym-envs-preview-3-parameters-torch]


# [start-isaac-gym-envs-preview-3-parameters-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_isaacgym_env_preview3

# load environment
env = load_isaacgym_env_preview3(task_name="Cartpole")
# [end-isaac-gym-envs-preview-3-parameters-jax]


# [start-isaac-gym-envs-preview-3-cli-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_isaacgym_env_preview3

# load environment
env = load_isaacgym_env_preview3()
# [end-isaac-gym-envs-preview-3-cli-torch]


# [start-isaac-gym-envs-preview-3-cli-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_isaacgym_env_preview3

# load environment
env = load_isaacgym_env_preview3()
# [end-isaac-gym-envs-preview-3-cli-jax]


# [start-isaac-gym-envs-preview-2-parameters-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_isaacgym_env_preview2

# load environment
env = load_isaacgym_env_preview2(task_name="Cartpole")
# [end-isaac-gym-envs-preview-2-parameters-torch]


# [start-isaac-gym-envs-preview-2-parameters-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_isaacgym_env_preview2

# load environment
env = load_isaacgym_env_preview2(task_name="Cartpole")
# [end-isaac-gym-envs-preview-2-parameters-jax]


# [start-isaac-gym-envs-preview-2-cli-torch]
# import the environment loader
from skrl.envs.loaders.torch import load_isaacgym_env_preview2

# load environment
env = load_isaacgym_env_preview2()
# [end-isaac-gym-envs-preview-2-cli-torch]


# [start-isaac-gym-envs-preview-2-cli-jax]
# import the environment loader
from skrl.envs.loaders.jax import load_isaacgym_env_preview2

# load environment
env = load_isaacgym_env_preview2()
# [end-isaac-gym-envs-preview-2-cli-jax]
