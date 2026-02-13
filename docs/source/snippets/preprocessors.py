# [start-running-standard-scaler-torch]
# import the preprocessor class
from skrl.resources.preprocessors.torch import RunningStandardScaler

cfg = AGENT_CFG()
# ...
cfg.observation_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
cfg.state_preprocessor = RunningStandardScaler
cfg.state_preprocessor_kwargs = {"size": env.state_space, "device": device}
cfg.value_preprocessor = RunningStandardScaler
cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
# [end-running-standard-scaler-torch]


# [start-running-standard-scaler-jax]
# import the preprocessor class
from skrl.resources.preprocessors.jax import RunningStandardScaler

cfg = AGENT_CFG()
# ...
cfg.observation_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
cfg.state_preprocessor = RunningStandardScaler
cfg.state_preprocessor_kwargs = {"size": env.state_space, "device": device}
cfg.value_preprocessor = RunningStandardScaler
cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
# [end-running-standard-scaler-jax]


# [start-running-standard-scaler-warp]
# import the preprocessor class
from skrl.resources.preprocessors.warp import RunningStandardScaler

cfg = AGENT_CFG()
# ...
cfg.observation_preprocessor = RunningStandardScaler
cfg.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
cfg.state_preprocessor = RunningStandardScaler
cfg.state_preprocessor_kwargs = {"size": env.state_space, "device": device}
cfg.value_preprocessor = RunningStandardScaler
cfg.value_preprocessor_kwargs = {"size": 1, "device": device}
# [end-running-standard-scaler-warp]
