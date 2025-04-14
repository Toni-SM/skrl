from typing import Any, Mapping, Type, Union

import copy

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
    def __init__(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any]) -> None:
        """Experiment runner

        Class that configures and instantiates skrl components to execute training/evaluation workflows in a few lines of code

        :param env: Environment to train on
        :param cfg: Runner configuration
        """
        self._env = env
        self._cfg = cfg

        # set random seed
        set_seed(self._cfg.get("seed", None))

        self._cfg["agent"]["rewards_shaper"] = None  # FIXME: avoid 'dictionary changed size during iteration'

        self._models = self._generate_models(self._env, copy.deepcopy(self._cfg))
        self._agent = self._generate_agent(self._env, copy.deepcopy(self._cfg), self._models)
        self._trainer = self._generate_trainer(self._env, copy.deepcopy(self._cfg), self._agent)

    @property
    def trainer(self) -> Trainer:
        """Trainer instance"""
        return self._trainer

    @property
    def agent(self) -> Agent:
        """Agent instance"""
        return self._agent

    @staticmethod
    def load_cfg_from_yaml(path: str) -> dict:
        """Load a runner configuration from a yaml file

        :param path: File path

        :return: Loaded configuration, or an empty dict if an error has occurred
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
        """Get skrl component (e.g.: agent, trainer, etc..) from string identifier

        :return: skrl component
        """
        component = None
        name = name.lower()
        # model
        if name == "gaussianmixin":
            from skrl.utils.model_instantiators.jax import gaussian_model as component
        elif name == "categoricalmixin":
            from skrl.utils.model_instantiators.jax import categorical_model as component
        elif name == "multicategoricalmixin":
            from skrl.utils.model_instantiators.jax import multicategorical_model as component
        elif name == "deterministicmixin":
            from skrl.utils.model_instantiators.jax import deterministic_model as component
        # memory
        elif name == "randommemory":
            from skrl.memories.jax import RandomMemory as component
        # agent
        elif name in ["a2c", "a2c_default_config"]:
            from skrl.agents.jax.a2c import A2C, A2C_DEFAULT_CONFIG

            component = A2C_DEFAULT_CONFIG if "default_config" in name else A2C
        elif name in ["cem", "cem_default_config"]:
            from skrl.agents.jax.cem import CEM, CEM_DEFAULT_CONFIG

            component = CEM_DEFAULT_CONFIG if "default_config" in name else CEM
        elif name in ["ddpg", "ddpg_default_config"]:
            from skrl.agents.jax.ddpg import DDPG, DDPG_DEFAULT_CONFIG

            component = DDPG_DEFAULT_CONFIG if "default_config" in name else DDPG
        elif name in ["ddqn", "ddqn_default_config"]:
            from skrl.agents.jax.dqn import DDQN, DDQN_DEFAULT_CONFIG

            component = DDQN_DEFAULT_CONFIG if "default_config" in name else DDQN
        elif name in ["dqn", "dqn_default_config"]:
            from skrl.agents.jax.dqn import DQN, DQN_DEFAULT_CONFIG

            component = DQN_DEFAULT_CONFIG if "default_config" in name else DQN
        elif name in ["ppo", "ppo_default_config"]:
            from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG

            component = PPO_DEFAULT_CONFIG if "default_config" in name else PPO
        elif name in ["rpo", "rpo_default_config"]:
            from skrl.agents.jax.rpo import RPO, RPO_DEFAULT_CONFIG

            component = RPO_DEFAULT_CONFIG if "default_config" in name else RPO
        elif name in ["sac", "sac_default_config"]:
            from skrl.agents.jax.sac import SAC, SAC_DEFAULT_CONFIG

            component = SAC_DEFAULT_CONFIG if "default_config" in name else SAC
        elif name in ["td3", "td3_default_config"]:
            from skrl.agents.jax.td3 import TD3, TD3_DEFAULT_CONFIG

            component = TD3_DEFAULT_CONFIG if "default_config" in name else TD3
        # multi-agent
        elif name in ["ippo", "ippo_default_config"]:
            from skrl.multi_agents.jax.ippo import IPPO, IPPO_DEFAULT_CONFIG

            component = IPPO_DEFAULT_CONFIG if "default_config" in name else IPPO
        elif name in ["mappo", "mappo_default_config"]:
            from skrl.multi_agents.jax.mappo import MAPPO, MAPPO_DEFAULT_CONFIG

            component = MAPPO_DEFAULT_CONFIG if "default_config" in name else MAPPO
        # trainer
        elif name == "sequentialtrainer":
            from skrl.trainers.jax import SequentialTrainer as component

        if component is None:
            raise ValueError(f"Unknown component '{name}' in runner cfg")
        return component

    def _process_cfg(self, cfg: dict) -> dict:
        """Convert simple types to skrl classes/components

        :param cfg: A configuration dictionary

        :return: Updated dictionary
        """
        _direct_eval = [
            "learning_rate_scheduler",
            "shared_state_preprocessor",
            "state_preprocessor",
            "value_preprocessor",
            "amp_state_preprocessor",
            "noise",
            "smooth_regularization_noise",
        ]

        def reward_shaper_function(scale):
            def reward_shaper(rewards, *args, **kwargs):
                return rewards * scale

            return reward_shaper

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
                    elif key in ["rewards_shaper_scale"]:
                        d["rewards_shaper"] = reward_shaper_function(value)
            return d

        return update_dict(copy.deepcopy(cfg))

    def _generate_models(
        self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any]
    ) -> Mapping[str, Mapping[str, Model]]:
        """Generate model instances according to the environment specification and the given config

        :param env: Wrapped environment
        :param cfg: A configuration dictionary

        :return: Model instances
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        # override cfg
        cfg["models"]["separate"] = True  # shared model is not supported in JAX

        agent_class = cfg.get("agent", {}).get("class", "").lower()

        # instantiate models
        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            models_cfg = _cfg.get("models")
            if not models_cfg:
                raise ValueError("No 'models' are defined in cfg")
            # get separate (non-shared) configuration and remove 'separate' key
            try:
                separate = models_cfg["separate"]
                del models_cfg["separate"]
            except KeyError:
                separate = True
                logger.warning("No 'separate' field defined in 'models' cfg. Defining it as True by default")
            # non-shared models
            if separate:
                for role in models_cfg:
                    # get instantiator function and remove 'class' key
                    model_class = models_cfg[role].get("class")
                    if not model_class:
                        raise ValueError(f"No 'class' field defined in 'models:{role}' cfg")
                    del models_cfg[role]["class"]
                    model_class = self._component(model_class)
                    # get specific spaces according to agent/model cfg
                    observation_space = observation_spaces[agent_id]
                    if agent_class == "mappo" and role == "value":
                        observation_space = state_spaces[agent_id]
                    if agent_class == "amp" and role == "discriminator":
                        try:
                            observation_space = env.amp_observation_space
                        except Exception as e:
                            logger.warning(
                                "Unable to get AMP space via 'env.amp_observation_space'. Using 'env.observation_space' instead"
                            )
                    # print model source
                    source = model_class(
                        observation_space=observation_space,
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
                model.init_state_dict(role)

        return models

    def _generate_agent(
        self,
        env: Union[Wrapper, MultiAgentEnvWrapper],
        cfg: Mapping[str, Any],
        models: Mapping[str, Mapping[str, Model]],
    ) -> Agent:
        """Generate agent instance according to the environment specification and the given config and models

        :param env: Wrapped environment
        :param cfg: A configuration dictionary
        :param models: Agent's model instances

        :return: Agent instances
        """
        multi_agent = isinstance(env, MultiAgentEnvWrapper)
        device = env.device
        num_envs = env.num_envs
        possible_agents = env.possible_agents if multi_agent else ["agent"]
        state_spaces = env.state_spaces if multi_agent else {"agent": env.state_space}
        observation_spaces = env.observation_spaces if multi_agent else {"agent": env.observation_space}
        action_spaces = env.action_spaces if multi_agent else {"agent": env.action_space}

        agent_class = cfg.get("agent", {}).get("class", "").lower()
        if not agent_class:
            raise ValueError(f"No 'class' field defined in 'agent' cfg")

        # check for memory configuration (backward compatibility)
        if not "memory" in cfg:
            logger.warning(
                "Deprecation warning: No 'memory' field defined in cfg. Using the default generated configuration"
            )
            cfg["memory"] = {"class": "RandomMemory", "memory_size": -1}
        # get memory class and remove 'class' field
        try:
            memory_class = self._component(cfg["memory"]["class"])
            del cfg["memory"]["class"]
        except KeyError:
            memory_class = self._component("RandomMemory")
            logger.warning("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")
        memories = {}
        # instantiate memory
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
        for agent_id in possible_agents:
            memories[agent_id] = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        # single-agent configuration and instantiation
        if agent_class in ["a2c", "cem", "ddpg", "ddqn", "dqn", "ppo", "rpo", "sac", "td3"]:
            agent_id = possible_agents[0]
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg.get("state_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("value_preprocessor_kwargs", {}).update({"size": 1, "device": device})
            if agent_cfg.get("exploration", {}).get("noise", None):
                agent_cfg["exploration"].get("noise_kwargs", {}).update({"device": device})
                agent_cfg["exploration"]["noise"] = agent_cfg["exploration"]["noise"](
                    **agent_cfg["exploration"].get("noise_kwargs", {})
                )
            if agent_cfg.get("smooth_regularization_noise", None):
                agent_cfg.get("smooth_regularization_noise_kwargs", {}).update({"device": device})
                agent_cfg["smooth_regularization_noise"] = agent_cfg["smooth_regularization_noise"](
                    **agent_cfg.get("smooth_regularization_noise_kwargs", {})
                )
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
        # multi-agent configuration and instantiation
        elif agent_class in ["ippo"]:
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update(
                {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "possible_agents": possible_agents,
            }
        elif agent_class in ["mappo"]:
            agent_cfg = self._component(f"{agent_class}_DEFAULT_CONFIG").copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update(
                {agent_id: {"size": observation_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg["shared_state_preprocessor_kwargs"].update(
                {agent_id: {"size": state_spaces[agent_id], "device": device} for agent_id in possible_agents}
            )
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "shared_observation_spaces": state_spaces,
                "possible_agents": possible_agents,
            }
        return self._component(agent_class)(cfg=agent_cfg, device=device, **agent_kwargs)

    def _generate_trainer(
        self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any], agent: Agent
    ) -> Trainer:
        """Generate trainer instance according to the environment specification and the given config and agent

        :param env: Wrapped environment
        :param cfg: A configuration dictionary
        :param agent: Agent's model instances

        :return: Trainer instances
        """
        # get trainer class and remove 'class' field
        try:
            trainer_class = self._component(cfg["trainer"]["class"])
            del cfg["trainer"]["class"]
        except KeyError:
            trainer_class = self._component("SequentialTrainer")
            logger.warning("No 'class' field defined in 'trainer' cfg. 'SequentialTrainer' will be used as default")
        # instantiate trainer
        return trainer_class(env=env, agents=agent, cfg=cfg["trainer"])

    def run(self, mode: str = "train") -> None:
        """Run the training/evaluation

        :param mode: Running mode: ``"train"`` for training or ``"eval"`` for evaluation (default: ``"train"``)

        :raises ValueError: The specified running mode is not valid
        """
        if mode == "train":
            self._trainer.train()
        elif mode == "eval":
            self._trainer.eval()
        else:
            raise ValueError(f"Unknown running mode: {mode}")
