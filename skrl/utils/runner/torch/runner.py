from __future__ import annotations

from typing import Any, Literal, Type

import copy
import dataclasses
import math  # noqa

from skrl import logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper, Wrapper
from skrl.models.torch import Model
from skrl.resources.noises.torch import GaussianNoise, OrnsteinUhlenbeckNoise  # noqa
from skrl.resources.preprocessors.torch import RunningStandardScaler  # noqa
from skrl.resources.schedulers.torch import KLAdaptiveLR  # noqa
from skrl.trainers.torch import Trainer
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
        self._verbose = verbose

        # check for configuration compatibility
        self._cfg = self._check_cfg_compatibility(copy.deepcopy(cfg))

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

    @staticmethod
    def _parse_multi_agent_models_cfg(models_cfg: dict, agent_id: str, possible_agents: list[str]) -> dict:
        """Get the models configuration of an agent (multi-agent).

        Models can be defined for all the agents (``models.<role>``) or for a specific agent
        (``models.<agent_id>.<role>``). Agent-specific models override (by role) the models defined for all
        the agents, and can also override the ``separate`` and ``single_forward_pass`` fields.

        :param models_cfg: Models configuration (``models`` field).
        :param agent_id: Agent id.
        :param possible_agents: Agent ids.

        :return: Models configuration of the agent.
        """
        common = {key: value for key, value in models_cfg.items() if key not in possible_agents}
        return {**common, **models_cfg.get(agent_id, {})}

    @staticmethod
    def _parse_multi_agent_kwargs(kwargs: dict, possible_agents: list[str], extra: dict[str, dict]) -> dict[str, dict]:
        """Get the agent-specific keyword arguments of a multi-agent setting, updated with extra ones.

        :param kwargs: Keyword arguments defined for all the agents, or per agent (if all agent ids are keys).
        :param possible_agents: Agent ids.
        :param extra: Extra keyword arguments by agent id.

        :raises ValueError: If the keyword arguments are defined for some agents only.

        :return: Keyword arguments by agent id.
        """
        if set(kwargs) & set(possible_agents) and not set(kwargs) >= set(possible_agents):
            raise ValueError(f"Specified keys ({set(kwargs)}) do not match possible agents ({set(possible_agents)})")
        per_agent = set(kwargs) >= set(possible_agents)
        return {
            agent_id: {**((kwargs[agent_id] or {}) if per_agent else kwargs), **extra[agent_id]}
            for agent_id in possible_agents
        }

    def _check_cfg_compatibility(self, cfg: dict) -> dict:
        """Check for configuration compatibility.

        :param cfg: Configuration dictionary to check for compatibility.

        :return: Updated dictionary.
        """
        # rename 'lambda' to 'gae_lambda'
        if "lambda" in cfg.get("agent", {}):
            logger.warning("The 'lambda' field in the configuration is deprecated. Use 'gae_lambda' instead")
            cfg["agent"]["gae_lambda"] = cfg["agent"]["lambda"]
            del cfg["agent"]["lambda"]
        # remove 'clip_predicted_values' redundant configuration by using 'value_clip'
        if "clip_predicted_values" in cfg.get("agent", {}):
            logger.warning(
                "The 'clip_predicted_values' field in the configuration is deprecated. "
                "Use a 'value_clip' value greater than 0 to clip the predicted values instead"
            )
            value_clip = cfg["agent"].get("value_clip", 0.2)
            cfg["agent"]["value_clip"] = value_clip if cfg["agent"]["clip_predicted_values"] else 0.0
            del cfg["agent"]["clip_predicted_values"]
        # replace `state_preprocessor` by `observation_preprocessor` if the latter is not defined
        if "state_preprocessor" in cfg.get("agent", {}):
            if "observation_preprocessor" not in cfg.get("agent", {}):
                logger.warning(
                    "The 'state_preprocessor' field in the configuration has been replaced by 'observation_preprocessor'. "
                    "If the 'state_preprocessor' definition is desired but the `observation_preprocessor` is not, "
                    "define the last one to None (null) to avoid the automatic replacement"
                )
                cfg["agent"]["observation_preprocessor"] = cfg["agent"]["state_preprocessor"]
                cfg["agent"]["observation_preprocessor_kwargs"] = cfg["agent"].get("state_preprocessor_kwargs")
                del cfg["agent"]["state_preprocessor"]
                if "state_preprocessor_kwargs" in cfg["agent"]:
                    del cfg["agent"]["state_preprocessor_kwargs"]
        # remove `shared_state_preprocessor` by using `state_preprocessor`
        if "shared_state_preprocessor" in cfg.get("agent", {}):
            logger.warning(
                "The 'shared_state_preprocessor' field in the configuration is deprecated. Use 'state_preprocessor' instead"
            )
            cfg["agent"]["state_preprocessor"] = cfg["agent"]["shared_state_preprocessor"]
            cfg["agent"]["state_preprocessor_kwargs"] = cfg["agent"].get("shared_state_preprocessor_kwargs")
            del cfg["agent"]["shared_state_preprocessor"]
            if "shared_state_preprocessor_kwargs" in cfg["agent"]:
                del cfg["agent"]["shared_state_preprocessor_kwargs"]
        return cfg

    def _component(self, name: str) -> Type:
        """Get skrl component (e.g.: agent, trainer, etc..) from string identifier.

        :return: skrl component.
        """
        from skrl.agents.torch.a2c import A2C, A2C_CFG
        from skrl.agents.torch.amp import AMP, AMP_CFG
        from skrl.agents.torch.cem import CEM, CEM_CFG
        from skrl.agents.torch.ddpg import DDPG, DDPG_CFG
        from skrl.agents.torch.ddqn import DDQN, DDQN_CFG
        from skrl.agents.torch.distillation import DISTILLATION_CFG, Distillation
        from skrl.agents.torch.dqn import DQN, DQN_CFG
        from skrl.agents.torch.ppo import PPO, PPO_CFG
        from skrl.agents.torch.rpo import RPO, RPO_CFG
        from skrl.agents.torch.sac import SAC, SAC_CFG
        from skrl.agents.torch.td3 import TD3, TD3_CFG
        from skrl.agents.torch.trpo import TRPO, TRPO_CFG
        from skrl.memories.torch import RandomMemory
        from skrl.multi_agents.torch.ippo import IPPO, IPPO_CFG
        from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_CFG
        from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg
        from skrl.utils.model_instantiators.torch import (
            categorical_model,
            deterministic_model,
            gaussian_model,
            multicategorical_model,
            multivariate_gaussian_model,
            shared_model,
        )

        component = {
            # models
            "gaussianmixin": gaussian_model,
            "categoricalmixin": categorical_model,
            "multicategoricalmixin": multicategorical_model,
            "deterministicmixin": deterministic_model,
            "multivariategaussianmixin": multivariate_gaussian_model,
            "shared": shared_model,
            # memories
            "randommemory": RandomMemory,
            # agents
            "a2c": A2C,
            "a2c_cfg": A2C_CFG,
            "amp": AMP,
            "amp_cfg": AMP_CFG,
            "cem": CEM,
            "cem_cfg": CEM_CFG,
            "ddpg": DDPG,
            "ddpg_cfg": DDPG_CFG,
            "ddqn": DDQN,
            "ddqn_cfg": DDQN_CFG,
            "distillation": Distillation,
            "distillation_cfg": DISTILLATION_CFG,
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
            "trpo": TRPO,
            "trpo_cfg": TRPO_CFG,
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

        def evaluate(value):
            # names can be defined per model (list) or per agent (dict)
            if isinstance(value, str):
                return eval(value)
            if isinstance(value, (list, tuple)):
                return type(value)(evaluate(item) for item in value)
            if isinstance(value, dict):
                return {key: evaluate(item) for key, item in value.items()}
            return value

        def update_dict(d):
            for key, value in d.items():
                if key in _direct_eval:
                    d[key] = evaluate(value)
                elif isinstance(value, dict):
                    update_dict(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
            return d

        cfg = update_dict(copy.deepcopy(cfg))
        if "class" in cfg:
            del cfg["class"]

        # materialize exploration scheduler
        if isinstance(cfg.get("exploration_scheduler"), str):
            cfg["exploration_scheduler"] = eval(f"lambda timestep, timesteps: {cfg['exploration_scheduler']}")
        # materialize rewards shaper
        if isinstance(cfg.get("rewards_shaper"), str):
            cfg["rewards_shaper"] = eval(f"lambda rewards, timestep, timesteps: {cfg['rewards_shaper']}")
        # backward compatibility: 'rewards_shaper_scale' (ignored if 'rewards_shaper' is defined)
        if "rewards_shaper_scale" in cfg:
            scale = cfg["rewards_shaper_scale"]
            if scale is not None and scale != 1.0:
                if cfg.get("rewards_shaper") is None:
                    cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * scale
                else:
                    logger.warning("Both 'rewards_shaper' and 'rewards_shaper_scale' are defined. Ignoring the scale")
            del cfg["rewards_shaper_scale"]

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
            if multi_agent:
                models_cfg = self._parse_multi_agent_models_cfg(models_cfg, agent_id, possible_agents)
            # get separate (non-shared) configuration and remove 'separate' key
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True
                logger.warning(
                    "The 'models.separate' field is not defined in the specified configuration. Falling back to True by default"
                )
            # get shared models' single forward-pass configuration and remove 'single_forward_pass' key
            single_forward_pass = models_cfg.pop("single_forward_pass", True)
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
                roles = list(models_cfg.keys())
                if len(roles) != 2:
                    raise ValueError(
                        "Runner currently only supports shared models, made up of exactly two models. "
                        "Set 'separate' field to True to create non-shared models for the given cfg"
                    )
                # get shared model structure and parameters
                structure = []
                parameters = []
                for role in roles:
                    # get instantiator function and remove 'class' key
                    model_structure = models_cfg[role].get("class")
                    if not model_structure:
                        raise ValueError(
                            f"The 'models.{role}.class' field is not defined in the specified configuration"
                        )
                    del models_cfg[role]["class"]
                    structure.append(model_structure)
                    parameters.append(self._process_cfg(models_cfg[role]))
                model_class = self._component("Shared")
                # print model source
                if self._verbose:
                    source = model_class(
                        observation_space=observation_spaces[agent_id],
                        state_space=state_spaces[agent_id],
                        action_space=action_spaces[agent_id],
                        device=device,
                        structure=structure,
                        roles=roles,
                        parameters=parameters,
                        single_forward_pass=single_forward_pass,
                        return_source=True,
                    )
                    print("==================================================")
                    print(f"Shared model (roles): {roles}")
                    print("==================================================\n")
                    print(source)
                    print("--------------------------------------------------")
                # instantiate model
                models[agent_id][roles[0]] = model_class(
                    observation_space=observation_spaces[agent_id],
                    state_space=state_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=structure,
                    roles=roles,
                    parameters=parameters,
                    single_forward_pass=single_forward_pass,
                )
                models[agent_id][roles[1]] = models[agent_id][roles[0]]

        # initialize lazy modules' parameters
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
        if agent_class in ["amp"]:
            agent_id = possible_agents[0]
            try:
                amp_observation_space = env.amp_observation_space
            except Exception as e:
                logger.warning(
                    "Unable to get AMP space via 'env.amp_observation_space'. Using 'env.observation_space' instead"
                )
                amp_observation_space = observation_spaces[agent_id]
            agent_cfg = dataclasses.asdict(self._component(f"{agent_class}_CFG")(**self._process_cfg(cfg["agent"])))
            agent_cfg.get("observation_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("state_preprocessor_kwargs", {}).update({"size": state_spaces[agent_id], "device": device})
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            agent_cfg.get("amp_observation_preprocessor_kwargs", {}).update(
                {"size": amp_observation_space, "device": device}
            )

            motion_dataset = None
            if "motion_dataset" in cfg:
                if "class" not in cfg["motion_dataset"]:
                    raise ValueError(f"The 'motion_dataset.class' field is not defined in the specified configuration")
                motion_dataset_class = self._component(cfg["motion_dataset"]["class"])
                motion_dataset = motion_dataset_class(device=device, **self._process_cfg(cfg["motion_dataset"]))
            reply_buffer = None
            if "reply_buffer" in cfg:
                if "class" not in cfg["reply_buffer"]:
                    raise ValueError(f"The 'reply_buffer.class' field is not defined in the specified configuration")
                reply_buffer_class = self._component(cfg["reply_buffer"]["class"])
                reply_buffer = reply_buffer_class(device=device, **self._process_cfg(cfg["reply_buffer"]))

            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "state_space": state_spaces[agent_id],
                "action_space": action_spaces[agent_id],
                "amp_observation_space": amp_observation_space,
                "motion_dataset": motion_dataset,
                "reply_buffer": reply_buffer,
                "collect_reference_motions": lambda num_samples: env.collect_reference_motions(num_samples),
            }
        elif agent_class in ["a2c", "cem", "ddpg", "ddqn", "distillation", "dqn", "ppo", "rpo", "sac", "td3", "trpo"]:
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
            for name, sizes in [
                ("observation_preprocessor_kwargs", observation_spaces),
                ("state_preprocessor_kwargs", state_spaces),
                ("value_preprocessor_kwargs", {agent_id: 1 for agent_id in possible_agents}),
            ]:
                agent_cfg[name] = self._parse_multi_agent_kwargs(
                    agent_cfg.get(name, {}),
                    possible_agents,
                    {agent_id: {"size": sizes[agent_id], "device": device} for agent_id in possible_agents},
                )
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
