"""
Notes for Isaac Sim 2022.2.1 or earlier (Python 3.7 environment):
  * Python 3.7 is only supported up to jax<=0.3.25.
    See: https://github.com/google/jax/blob/main/CHANGELOG.md#jaxlib-041-dec-13-2022.
  * Builds for jaxlib<=0.3.25 are only available up to NVIDIA CUDA 11 and cuDNN 8.2 versions.
    See: https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    and search for `cuda11/jaxlib-0.3.25+cuda11.cudnn82-cp37-cp37m-manylinux2014_x86_64.whl`.
  * The `jax.Device = jax.xla.Device` statement is required by skrl to support jax<0.4.3.
  * Models require overloading the `__hash__` method to avoid "TypeError: Failed to hash Flax Module".
"""

import threading

import flax.linen as nn
import jax
import jax.numpy as jnp


jax.Device = jax.xla.Device  # for Isaac Sim 2022.2.1 or earlier

# import the skrl components to build the RL system
from skrl import config
from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.jax import load_omniverse_isaacgym_env
from skrl.envs.wrappers.jax import wrap_env
from skrl.memories.jax import RandomMemory
from skrl.models.jax import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.resources.schedulers.jax import KLAdaptiveRL
from skrl.trainers.jax import SequentialTrainer
from skrl.utils import set_seed


config.jax.backend = "jax"  # or "numpy"


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device=None, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

    def __hash__(self):  # for Isaac Sim 2022.2.1 or earlier
        return id(self)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.elu(nn.Dense(256)(inputs["states"]))
        x = nn.elu(nn.Dense(128)(x))
        x = nn.elu(nn.Dense(64)(x))
        x = nn.Dense(self.num_actions)(x)
        log_std = self.param("log_std", lambda _: jnp.zeros(self.num_actions))
        return x, log_std, {}

class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device=None, clip_actions=False, **kwargs):
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self, clip_actions)

    def __hash__(self):  # for Isaac Sim 2022.2.1 or earlier
        return id(self)

    @nn.compact  # marks the given module method allowing inlined submodules
    def __call__(self, inputs, role):
        x = nn.elu(nn.Dense(256)(inputs["states"]))
        x = nn.elu(nn.Dense(128)(x))
        x = nn.elu(nn.Dense(64)(x))
        x = nn.Dense(1)(x)
        return x, {}


# load and wrap the multi-threaded Omniverse Isaac Gym environment
env = load_omniverse_isaacgym_env(task_name="Ant", multi_threaded=True, timeout=30)
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=16, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Policy(env.observation_space, env.action_space, device)
models["value"] = Value(env.observation_space, env.action_space, device)

# instantiate models' state dict
for role, model in models.items():
    model.init_state_dict(role)


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 16  # memory_size
cfg["learning_epochs"] = 4
cfg["mini_batches"] = 2  # 16 * 4096 / 32768
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 3e-4
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = lambda rewards, timestep, timesteps: rewards * 0.01
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 40
cfg["experiment"]["checkpoint_interval"] = 400
cfg["experiment"]["directory"] = "runs/jax/Ant"

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 8000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training in a separate thread
threading.Thread(target=trainer.train).start()


# run the simulation in the main thread
env.run()
