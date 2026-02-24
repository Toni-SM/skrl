# [start-3rd-party-torch]
# import the scheduler class
from torch.optim.lr_scheduler import StepLR

cfg = AGENT_CFG()
# ...
cfg.learning_rate_scheduler = StepLR
cfg.learning_rate_scheduler_kwargs = {"step_size": 1, "gamma": 0.9}
# [end-3rd-party-torch]


# [start-native-torch]
# import the scheduler class
from skrl.resources.schedulers.torch import KLAdaptiveLR

cfg = AGENT_CFG()
# ...
cfg.learning_rate_scheduler = KLAdaptiveLR
cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}
# [end-native-torch]


# [start-3rd-party-jax]
# import the scheduler function
from optax import constant_schedule

cfg = AGENT_CFG()
# ...
cfg.learning_rate_scheduler = constant_schedule
cfg.learning_rate_scheduler_kwargs = {"value": 1e-4}
# [end-3rd-party-jax]


# [start-native-jax]
# import the scheduler function
from skrl.resources.schedulers.jax import KLAdaptiveLR  # or kl_adaptive (Optax style)

cfg = AGENT_CFG()
# ...
cfg.learning_rate_scheduler = KLAdaptiveLR
cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}
# [end-native-jax]


# [start-native-warp]
# import the scheduler function
from skrl.resources.schedulers.warp import KLAdaptiveLR  # or kl_adaptive

cfg = AGENT_CFG()
# ...
cfg.learning_rate_scheduler = KLAdaptiveLR
cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}
# [end-native-warp]

# =============================================================================

# [start-scheduler-kl-adaptive-torch]
# import the scheduler class
from skrl.resources.schedulers.torch import KLAdaptiveLR

cfg = AGENT_CFG()
# ...
cfg.learning_rate_scheduler = KLAdaptiveLR
cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}
# [end-scheduler-kl-adaptive-torch]


# [start-scheduler-kl-adaptive-jax]
# import the scheduler function
from skrl.resources.schedulers.jax import KLAdaptiveLR  # or kl_adaptive (Optax style)

cfg = AGENT_CFG()
# ...
cfg.learning_rate_scheduler = KLAdaptiveLR
cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}
# [end-scheduler-kl-adaptive-jax]


# [start-scheduler-kl-adaptive-warp]
# import the scheduler function
from skrl.resources.schedulers.warp import KLAdaptiveLR  # or kl_adaptive

cfg = AGENT_CFG()
# ...
cfg.learning_rate_scheduler = KLAdaptiveLR
cfg.learning_rate_scheduler_kwargs = {"kl_threshold": 0.01}
# [end-scheduler-kl-adaptive-warp]
