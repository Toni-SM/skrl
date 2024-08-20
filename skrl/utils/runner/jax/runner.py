from typing import Any, Mapping, Type, Union

import copy

from skrl import config, logger
from skrl.agents.jax import Agent
from skrl.agents.jax.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.jax import MultiAgentEnvWrapper, Wrapper
from skrl.memories.jax import RandomMemory
from skrl.models.jax import Model
from skrl.multi_agents.jax.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.multi_agents.jax.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.jax import RunningStandardScaler
from skrl.resources.schedulers.jax import KLAdaptiveLR
from skrl.trainers.jax import SequentialTrainer, Trainer
from skrl.utils import set_seed
from skrl.utils.model_instantiators.jax import Shape, deterministic_model, gaussian_model


class Runner:
    def __init__(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any]) -> None:
        """Experiment runner

        Class that configures and instantiates skrl components to execute training/evaluation workflows in a few lines of code

        .. note::

            This class encapsulates, and greatly simplifies, the definitions and instantiations needed to execute RL tasks.
            However, such simplification hides and makes difficult the modification and readability of the code (models, agents, etc.).

            For more control and readability over the RL system setup refer to the *Examples* section's training scripts (recommended!)

        :param env: Environment to train on
        :param cfg: Runner configuration
        """
        self._env = env
        self._cfg = cfg

        self._cfg["agent"]["rewards_shaper"] = None  # FIXME: avoid 'dictionary changed size during iteration'

        self._class_mapping = {
            # model
            "gaussianmixin": gaussian_model,
            "deterministicmixin": deterministic_model,
            "shared": None,
            # memory
            "randommemory": RandomMemory,
            # agent
            "ppo": PPO,
            "ippo": IPPO,
            "mappo": MAPPO,
            # trainer
            "sequentialtrainer": SequentialTrainer,
        }

        # set backend
        config.jax.backend = self._cfg.get("backend", "jax")

        # set random seed
        set_seed(self._cfg.get("seed", None))

        self._models = self._generate_models(self._env, copy.deepcopy(self._cfg))
        self._agent = self._generate_agent(self._env, copy.deepcopy(self._cfg), self._models)
        self._trainer = self._generate_trainer(self._env, copy.deepcopy(self._cfg), self._agent)

    @property
    def trainer(self) -> Trainer:
        """Trainer instance
        """
        return self._trainer

    @property
    def agent(self) -> Agent:
        """Agent instance
        """
        return self._agent

    def _class(self, value: str) -> Type:
        """Get skrl component class (e.g.: agent, trainer, etc..) from string identifier

        :return: skrl component class
        """
        if value.lower() in self._class_mapping:
            return self._class_mapping[value.lower()]
        raise ValueError(f"Unknown class '{value}' in runner cfg")

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
            "input_shape",
            "output_shape",
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
                        d[key] = eval(value)
                    elif key.endswith("_kwargs"):
                        d[key] = value if value is not None else {}
                    elif key in ["rewards_shaper_scale"]:
                        d["rewards_shaper"] = reward_shaper_function(value)

            return d

        return update_dict(copy.deepcopy(cfg))

    def _generate_models(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any]) -> Mapping[str, Mapping[str, Model]]:
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

        # instantiate models
        models = {}
        for agent_id in possible_agents:
            _cfg = copy.deepcopy(cfg)
            models[agent_id] = {}
            # non-shared models
            if _cfg["models"]["separate"]:
                # get instantiator function and remove 'class' field
                try:
                    model_class = self._class(_cfg["models"]["policy"]["class"])
                    del _cfg["models"]["policy"]["class"]
                except KeyError:
                    model_class = self._class("GaussianMixin")
                    logger.warning("No 'class' field defined in 'models:policy' cfg. 'GaussianMixin' will be used as default")
                # instantiate model
                models[agent_id]["policy"] = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    **self._process_cfg(_cfg["models"]["policy"]),
                )
                # get instantiator function and remove 'class' field
                try:
                    model_class = self._class(_cfg["models"]["value"]["class"])
                    del _cfg["models"]["value"]["class"]
                except KeyError:
                    model_class = self._class("DeterministicMixin")
                    logger.warning("No 'class' field defined in 'models:value' cfg. 'DeterministicMixin' will be used as default")
                # instantiate model
                models[agent_id]["value"] = model_class(
                    observation_space=(state_spaces if self._class(_cfg["agent"]["class"]) in [MAPPO] else observation_spaces)[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    **self._process_cfg(_cfg["models"]["value"]),
                )
            # shared models
            else:
                # remove 'class' field
                try:
                    del _cfg["models"]["policy"]["class"]
                except KeyError:
                    logger.warning("No 'class' field defined in 'models:policy' cfg. 'GaussianMixin' will be used as default")
                try:
                    del _cfg["models"]["value"]["class"]
                except KeyError:
                    logger.warning("No 'class' field defined in 'models:value' cfg. 'DeterministicMixin' will be used as default")
                # instantiate model
                model_class = self._class("Shared")
                models[agent_id]["policy"] = model_class(
                    observation_space=observation_spaces[agent_id],
                    action_space=action_spaces[agent_id],
                    device=device,
                    structure=None,
                    roles=["policy", "value"],
                    parameters=[
                        self._process_cfg(_cfg["models"]["policy"]),
                        self._process_cfg(_cfg["models"]["value"]),
                    ],
                )
                models[agent_id]["value"] = models[agent_id]["policy"]

        # instantiate models' state dict
        for role, model in models.items():
            model.init_state_dict(role)

        return models

    def _generate_agent(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any], models: Mapping[str, Mapping[str, Model]]) -> Agent:
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

        # instantiate memories
        memories = {}
        # get memory class and remove 'class' field
        try:
            memory_class = self._class(cfg["memory"]["class"])
            del cfg["memory"]["class"]
        except KeyError:
            memory_class = self._class("RandomMemory")
            logger.warning("No 'class' field defined in 'memory' cfg. 'RandomMemory' will be used as default")
        # instantiate memory
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]  # memory_size is the agent's number of rollouts
        for agent_id in possible_agents:
            memories[agent_id] = memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))

        # instantiate agent
        try:
            agent_class = self._class(cfg["agent"]["class"])
            del cfg["agent"]["class"]
        except KeyError:
            agent_class = self._class("PPO")
            logger.warning("No 'class' field defined in 'agent' cfg. 'PPO' will be used as default")
        # single-agent configuration and instantiation
        if agent_class in [PPO]:
            agent_id = possible_agents[0]
            agent_cfg = PPO_DEFAULT_CONFIG.copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update({"size": observation_spaces[agent_id], "device": device})
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
        # multi-agent configuration and instantiation
        elif agent_class in [IPPO]:
            agent_cfg = IPPO_DEFAULT_CONFIG.copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update({
                agent_id: {"size": observation_spaces[agent_id], "device": device}
                for agent_id in possible_agents
            })
            agent_cfg["value_preprocessor_kwargs"].update({"size": 1, "device": device})
            agent_kwargs = {
                "models": models,
                "memories": memories,
                "observation_spaces": observation_spaces,
                "action_spaces": action_spaces,
                "possible_agents": possible_agents,
            }
        elif agent_class in [MAPPO]:
            agent_cfg = MAPPO_DEFAULT_CONFIG.copy()
            agent_cfg.update(self._process_cfg(cfg["agent"]))
            agent_cfg["state_preprocessor_kwargs"].update({
                agent_id: {"size": observation_spaces[agent_id], "device": device}
                for agent_id in possible_agents
            })
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
        return agent_class(cfg=agent_cfg, device=device, **agent_kwargs)

    def _generate_trainer(self, env: Union[Wrapper, MultiAgentEnvWrapper], cfg: Mapping[str, Any], agent: Agent) -> Trainer:
        """Generate trainer instance according to the environment specification and the given config and agent

        :param env: Wrapped environment
        :param cfg: A configuration dictionary
        :param agent: Agent's model instances

        :return: Trainer instances
        """
        # get trainer class and remove 'class' field
        try:
            trainer_class = self._class(cfg["trainer"]["class"])
            del cfg["trainer"]["class"]
        except KeyError:
            trainer_class = self._class("SequentialTrainer")
            logger.warning("No 'class' field defined in 'trainer' cfg. 'SequentialTrainer' will be used as default")
        # instantiate trainer
        return trainer_class(env=env, agents=agent, cfg=cfg["trainer"])

    def train(self) -> None:
        """Train the agent
        """
        self._trainer.train()

    def eval(self) -> None:
        """Evaluate agent
        """
        self._trainer.eval()
