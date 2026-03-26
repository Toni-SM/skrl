from __future__ import annotations

from typing import Any, Literal, Type

import copy
import dataclasses
import math  # noqa

from skrl import logger
from skrl.agents.jax import Agent
from skrl.envs.wrappers.jax import MultiAgentEnvWrapper, Wrapper
from skrl.models.jax import Model
from skrl.resources.noises.jax import GaussianNoise, OrnsteinUhlenbeckNoise  # noqa
from skrl.resources.preprocessors.jax import RunningStandardScaler  # noqa
from skrl.resources.schedulers.jax import KLAdaptiveLR  # noqa
from skrl.trainers.jax import Trainer
from skrl.utils import set_seed


class Runner:
    def __init__(self, env: Wrapper | MultiAgentEnvWrapper, cfg: dict[str, Any], *, verbose: bool = False) -> None:
        """Experiment runner.

        Configure and instantiate skrl components to execute training/evaluation workflows in a few lines of code.

        :param env: Environment to train on.
        :param cfg: Runner configuration.
        :param verbose: Whether to print extra information about the setup.
        """
        self._env = env
        self._cfg = cfg
        self._verbose = verbose

        # set random seed
        set_seed(self._cfg.get("seed", None))

        self._models = self._generate_models(self._env, copy.deepcopy(self._cfg))
        self._agent = self._generate_agent(self._env, copy.deepcopy(self._cfg), self._models)
        self._trainer = self._generate_trainer(self._env, copy.deepcopy(self._cfg), self._agent)

    @property
    def trainer(self) -> Trainer:
        """Trainer instance."""
        return self._trainer

    @property
    def agent(self) -> Agent:
        """Agent instance."""
        return self._agent

    @staticmethod
    def load_cfg_from_yaml(path: str) -> dict:
        """Load a runner configuration from a yaml file.

        :param path: File path.

        :return: Loaded configuration, or an empty dict if an error has occurred.
        """
        try:
            import yaml
        except Exception as e:
            logger.error(f"{e}. Install PyYAML with 'pip install pyyaml'")
            return {}

        try:
            with open(path) as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Loading yaml error: {e}")
            return {}

    def _component(self, name: str) -> Type:
        """Get skrl component (e.g.: agent, trainer, etc..) from string identifier.

        :return: skrl component.
        """
        from skrl.agents.jax.a2c import A2C, A2C_CFG
        from skrl.agents.jax.cem import CEM, CEM_CFG
        from skrl.agents.jax.ddpg import DDPG, DDPG_CFG
        from skrl.agents.jax.ddqn import DDQN, DDQN_CFG
        from skrl.agents.jax.dqn import DQN, DQN_CFG
        from skrl.agents.jax.ppo import PPO, PPO_CFG
        from skrl.agents.jax.rpo import RPO, RPO_CFG
        from skrl.agents.jax.sac import SAC, SAC_CFG
        from skrl.agents.jax.td3 import TD3, TD3_CFG
        from skrl.memories.jax import RandomMemory
        from skrl.multi_agents.jax.ippo import IPPO, IPPO_CFG
        from skrl.multi_agents.jax.mappo import MAPPO, MAPPO_CFG
        from skrl.trainers.jax import SequentialTrainer, SequentialTrainerCfg
        from skrl.utils.model_instantiators.jax import (
            categorical_model,
            deterministic_model,
            gaussian_model,
            multicategorical_model,
        )

        component = {
            # models
            "gaussianmixin": gaussian_model,
            "categoricalmixin": categorical_model,
            "multicategoricalmixin": multicategorical_model,
            "deterministicmixin": deterministic_model,
            # memories
            "randommemory": RandomMemory,
            # agents
            "a2c": A2C,
            "a2c_cfg": A2C_CFG,
            "cem": CEM,
            "cem_cfg": CEM_CFG,
            "ddpg": DDPG,
            "ddpg_cfg": DDPG_CFG,
            "ddqn": DDQN,
            "ddqn_cfg": DDQN_CFG,
            "dqn": DQN,
            "dqn_cfg": DQN_CFG,
            "ppo": PPO,
            "ppo_cfg": PPO_CFG,
            "rpo": RPO,
            "rpo_cfg": RPO_CFG,
            "sac": SAC,
            "sac_cfg": SAC_CFG,
            "td3": TD3,
            "td3_cfg": TD3_CFG,
            # multi-agents
            "ippo": IPPO,
            "ippo_cfg": IPPO_CFG,
            "mappo": MAPPO,
            "mappo_cfg": MAPPO_CFG,
            # trainers
            "sequentialtrainer": SequentialTrainer,
            "sequentialtrainer_cfg": SequentialTrainerCfg,
        }.get(name.lower())

        if component is None:
            raise ValueError(f"Component '{name}' is not supported in the runner cfg")
        return component

    def _process_cfg(self, cfg: dict) -> dict:
        """Convert simple types to skrl classes/components.

        :param cfg: A configuration dictionary.

        :return: Updated dictionary.
        """
        _direct_eval = [
            "learning_rate_scheduler",
            "observation_preprocessor",
            "state_preprocessor",
            "value_preprocessor",
            "amp_observation_preprocessor",
            "exploration_noise",
            "smooth_regularization_noise",
        ]

        def update_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    update_dict(value)
                else:
                    if key in _direct_eval:
                        if isinstance(value, str):
                            d[key] = eval(value)
                    elif key.endswith("_kwargs"):
                        d[key] = value if value is not None else {}
            return d

        cfg = update_dict(copy.deepcopy(cfg))
        if "class" in cfg:
            del cfg["class"]

        # materialize exploration scheduler
        if "exploration_scheduler" in cfg:
            cfg["exploration_scheduler"] = eval(f"lambda timestep, timesteps: {cfg['exploration_scheduler']}")
        # materialize rewards shaper
        if "rewards_shaper_scale" in cfg:
            scale = cfg["rewards_shaper_scale"]
            if scale is not None and scale != 1.0:
                cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * scale
            del cfg["rewards_shaper_scale"]

        # backward compatibility
        if "lambda" in cfg:
            logger.warning("The 'lambda' field in the specified configuration is deprecated. Use 'gae_lambda' instead")
            cfg["gae_lambda"] = cfg["lambda"]
            del cfg["lambda"]
        if "clip_predicted_values" in cfg:
            logger.warning(
                "The 'clip_predicted_values' field in the specified configuration is deprecated. "
                "Define a 'value_clip' value greater than 0 to clip the predicted values instead"
            )
            cfg["value_clip"] = cfg["value_clip"] if cfg["clip_predicted_values"] else 0.0
            del cfg["clip_predicted_values"]

        return cfg

    def _generate_models(self, env: Wrapper | MultiAgentEnvWrapper, cfg: dict[str, Any]) -> dict[str, dict[str, Model]]:
        """Generate model instances according to the environment specification and the given config.

        :param env: Wrapped environment.
        :param cfg: A configuration dictionary.

        :return: Model instances.
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        # override cfg
        if cfg.get("models", {}).get("separate") is not None:
            cfg["models"]["separate"] = True  # shared model is not supported in JAX

        agent_class = cfg.get("agent", {}).get("class")
        if not agent_class:
            raise ValueError(f"The 'agent.class' field is not defined in the specified configuration")
        agent_class = agent_class.lower()

        # instantiate models
        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("The 'models' field is not defined in the specified configuration")
            # get separate (non-shared) configuration and remove 'separate' key
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True
                logger.warning(
                    "The 'models.separate' field is not defined in the specified configuration. Falling back to True by default"
                )
            # non-shared models
            if separate:
                for role in models_cfg:
                    # get instantiator function and remove 'class' key
                    model_class = models_cfg[role].get("class")
                    if not model_class:
                        raise ValueError(
                            f"The 'models.{role}.class' field is not defined in the specified configuration"
                        )
                    del models_cfg[role]["class"]
                    model_class = self._component(model_class)
                    # get specific spaces according to agent/model cfg
                    observation_space = observation_spaces[agent_id]
                    if agent_class == "amp" and role == "discriminator":
                        try:
                            observation_space = env.amp_observation_space
                        except Exception as e:
                            logger.warning(
                                "Unable to get AMP space via 'env.amp_observation_space'. Using 'env.observation_space' instead"
                            )
                    # print model source
                    if self._verbose:
                        source = model_class(
                            observation_space=observation_space,
                            state_space=state_spaces[agent_id],
                            action_space=action_spaces[agent_id],
                            device=device,
                            **self._process_cfg(models_cfg[role]),
                            return_source=True,
                        )
                        print("==================================================")
                        print(f"Model (role): {role}")
                        print("==================================================\n")
                        print(source)
                        print("--------------------------------------------------")
                    # instantiate model
                    models[agent_id][role] = model_class(
                        observation_space=observation_space,
                        state_space=state_spaces[agent_id],
                        action_space=action_spaces[agent_id],
                        device=device,
                        **self._process_cfg(models_cfg[role]),
                    )
            # shared models
            else:
                raise NotImplementedError

        # initialize models' state dict
        for agent_id in possible_agents:
            for role, model in models[agent_id].items():
                model.init_state_dict(role=role)

        return models

    def _generate_agent(
        self,
        env: Wrapper | MultiAgentEnvWrapper,
        cfg: dict[str, Any],
        models: dict[str, dict[str, Model]],
    ) -> Agent:
        """Generate agent instance according to the environment specification and the given config and models.

        :param env: Wrapped environment.
        :param cfg: A configuration dictionary.
        :param models: Agent's model instances.

        :return: Agent instances.
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        num_envs = env.num_envs
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        # override cfg
        if cfg.get("agent", {}).get("mixed_precision") is not None:
            del cfg["agent"]["mixed_precision"]  # mixed precision is not supported in JAX

        # get agent class
        if "agent" not in cfg:
            raise ValueError(f"The 'agent' field is not defined in the specified configuration")
        if "class" not in cfg["agent"]:
            raise ValueError(f"The 'agent.class' field is not defined in the specified configuration")
        agent_class = cfg["agent"]["class"].lower()

        # get memory class
        if "memory" not in cfg:
            raise ValueError(f"The 'memory' field is not defined in the specified configuration")
        if "class" not in cfg["memory"]:
            raise ValueError(f"The 'memory.class' field is not defined in the specified configuration")
        memory_class = self._component(cfg["memory"]["class"])
        # instantiate memory
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
        memories = {
            agent_id: memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))
            for agent_id in possible_agents
        }

        # single-agent configuration and instantiation
        if agent_class in ["a2c", "cem", "ddpg", "ddqn", "dqn", "ppo", "rpo", "sac", "td3"]:
            agent_id = possible_agents[0]
            agent_cfg = dataclasses.asdict(self._component(f"{agent_class}_CFG")(**self._process_cfg(cfg["agent"])))
            agent_cfg.get("observation_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("state_preprocessor_kwargs", {}).update({"size": state_spaces[agent_id], "device": device})
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            agent_cfg.get("exploration_noise_kwargs", {}).update({"device": device})
            agent_cfg.get("smooth_regularization_noise_kwargs", {}).update({"device": device})
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "state_space": state_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
        # multi-agent configuration and instantiation
        elif agent_class in ["ippo", "mappo"]:
            agent_cfg = dataclasses.asdict(self._component(f"{agent_class}_CFG")(**self._process_cfg(cfg["agent"])))
            agent_cfg.get("observation_preprocessor_kwargs", {}).update(
                {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg.get("state_preprocessor_kwargs", {}).update(
                {agent_id: {"size": state_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "state_spaces": state_spaces,
                "action_spaces": action_spaces,
                "possible_agents": possible_agents,
            }
        return self._component(agent_class)(cfg=agent_cfg, device=device, **agent_kwargs)

    def _generate_trainer(self, env: Wrapper | MultiAgentEnvWrapper, cfg: dict[str, Any], agent: Agent) -> Trainer:
        """Generate trainer instance according to the environment specification and the given config and agent.

        :param env: Wrapped environment.
        :param cfg: A configuration dictionary.
        :param agent: Agent's model instances.

        :return: Trainer instances.
        """
        # get trainer class
        if "trainer" not in cfg:
            raise ValueError(f"The 'trainer' field is not defined in the specified configuration")
        if "class" not in cfg["trainer"]:
            raise ValueError(f"The 'trainer.class' field is not defined in the specified configuration")
        trainer_class = cfg["trainer"]["class"].lower()
        # instantiate trainer
        trainer_cfg = self._component(f"{trainer_class}_CFG")(**self._process_cfg(cfg["trainer"]))
        return self._component(trainer_class)(env=env, agents=agent, cfg=trainer_cfg)

    def run(self, mode: Literal["train", "eval"] = "train") -> None:
        """Run the training/evaluation.

        :param mode: Running mode: ``"train"`` for training or ``"eval"`` for evaluation.

        :raises ValueError: The specified running mode is not valid.
        """
        if mode == "train":
            self._trainer.train()
        elif mode == "eval":
            self._trainer.eval()
        else:
            raise ValueError(f"Unknown running mode: {mode}")
