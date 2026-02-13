# [torch-start-gaussian]
# import the noise class
from skrl.resources.noises.torch import GaussianNoise

cfg = AGENT_CFG()
# ...
cfg.exploration_noise = GaussianNoise
cfg.exploration_noise_kwargs = {"mean": 0.0, "std": 0.1, "device": device}
# [torch-end-gaussian]


# [jax-start-gaussian]
# import the noise class
from skrl.resources.noises.jax import GaussianNoise

cfg = AGENT_CFG()
# ...
cfg.exploration_noise = GaussianNoise
cfg.exploration_noise_kwargs = {"mean": 0.0, "std": 0.1, "device": device}
# [jax-end-gaussian]


# [warp-start-gaussian]
# import the noise class
from skrl.resources.noises.warp import GaussianNoise

cfg = AGENT_CFG()
# ...
cfg.exploration_noise = GaussianNoise
cfg.exploration_noise_kwargs = {"mean": 0.0, "std": 0.1, "device": device}
# [warp-end-gaussian]

# =============================================================================

# [torch-start-ornstein-uhlenbeck]
# import the noise class
from skrl.resources.noises.torch import OrnsteinUhlenbeckNoise

cfg = AGENT_CFG()
# ...
cfg.exploration_noise = OrnsteinUhlenbeckNoise
cfg.exploration_noise_kwargs = {"theta": 0.15, "sigma": 0.1, "base_scale": 0.5, "device": device}
# [torch-end-ornstein-uhlenbeck]


# [jax-start-ornstein-uhlenbeck]
# import the noise class
from skrl.resources.noises.jax import OrnsteinUhlenbeckNoise

cfg = AGENT_CFG()
# ...
cfg.exploration_noise = OrnsteinUhlenbeckNoise
cfg.exploration_noise_kwargs = {"theta": 0.15, "sigma": 0.1, "base_scale": 0.5, "device": device}
# [jax-end-ornstein-uhlenbeck]


# [warp-start-ornstein-uhlenbeck]
# import the noise class
from skrl.resources.noises.warp import OrnsteinUhlenbeckNoise

cfg = AGENT_CFG()
# ...
cfg.exploration_noise = OrnsteinUhlenbeckNoise
cfg.exploration_noise_kwargs = {"theta": 0.15, "sigma": 0.1, "base_scale": 0.5, "device": device}
# [warp-end-ornstein-uhlenbeck]
