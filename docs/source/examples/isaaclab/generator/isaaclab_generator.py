import os
os.environ["HEADLESS"] = "1"

from omni.isaac.lab.app import AppLauncher

app_launcher = AppLauncher()
simulation_app = app_launcher.app

import math
import gymnasium as gym
import importlib
import inspect
import subprocess
from jinja2 import Template
from prettytable import PrettyTable

from omni.isaac.lab.utils import class_to_dict
from omni.isaac.lab_tasks.utils import load_cfg_from_registry


class Config:
    def __init__(self, library: str) -> None:
        self.library = library

        self.cfg = {}
        self.path = ""
        self.filename = ""
        self.valid = False
        self.env_name = None
        self.num_envs = None
        self.algorithm = None

        self._templates = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

    def _parse_rl_games(self, cfg: dict) -> None:
        self.cfg = cfg
        self.cfg["metadata"] = {"num_envs": self.num_envs, "task": self.env_name}
        # check configuration
        # algorithm
        algorithm = self.cfg["params"]["algo"]["name"]
        assert algorithm.lower() == "a2c_continuous", f"Unknown rl_games agent: {algorithm}"
        assert self.cfg["params"]["config"]["ppo"], f"rl_games's PPO config is set to False"
        self.algorithm = "ppo"
        # model
        model = self.cfg["params"]["model"]["name"]
        assert model.lower() == "continuous_a2c_logstd", f"Unknown rl_games model: {model}"
        mini_batches = (
            self.cfg["params"]["config"]["horizon_length"]
            * self.num_envs
            / self.cfg["params"]["config"]["minibatch_size"]
        )
        # rnn/cnn
        assert "central_value_config" not in self.cfg["params"]["config"], f"rl_games's config includes central_value_config"
        assert "rnn" not in self.cfg["params"]["network"], f"rl_games's network includes RNN"
        assert "cnn" not in self.cfg["params"]["network"], f"rl_games's network includes CNN"
        # mini batches
        if isinstance(mini_batches, float):
            assert mini_batches.is_integer(), (
                "Invalid rl_games configuration: mini_batches is not integer:"
                f" {self.cfg['params']['config']['horizon_length']} * {self.num_envs} /"
                f" {self.cfg['params']['config']['minibatch_size']} -> {mini_batches}"
            )

    def _parse_rsl_rl(self, cfg: object) -> None:
        self.cfg = class_to_dict(cfg)
        self.cfg["metadata"] = {"num_envs": self.num_envs, "task": self.env_name}
        # check configuration
        algorithm = self.cfg["algorithm"]["class_name"]
        assert algorithm.lower() == "ppo", f"Unknown rsl_rl agent: {algorithm}"
        self.algorithm = "ppo"
        # convert config
        self.cfg["policy"]["init_noise_std__converted"] = math.log(self.cfg["policy"]["init_noise_std"])

    def _parse_skrl(self, cfg: dict) -> None:
        self.cfg = cfg
        self.cfg["metadata"] = {"num_envs": self.num_envs, "task": self.env_name}
        self.algorithm = self.cfg["agent"].get("class", "PPO").lower()

    def parse(self, entry: str, env_name: str, num_envs: int) -> "Config":
        self.env_name = env_name
        self.num_envs = num_envs
        try:
            # load cfg from registered entry
            cfg = load_cfg_from_registry(env_name, entry)
            # get path
            cfg_entry_point = gym.spec(env_name).kwargs.get(entry)
            if isinstance(cfg_entry_point, str):
                mod_name, file_name = cfg_entry_point.split(":")
                if "rsl_rl" in mod_name:
                    file_name = mod_name.split(".")[-1] + ".py"
                    mod_name = ".".join(mod_name.split(".")[:-1])
                self.filename = file_name
                mod_path = os.path.dirname(importlib.import_module(mod_name).__file__)
                self.path = os.path.join(mod_path, file_name)
            elif callable(cfg_entry_point):
                self.path = inspect.getfile(cfg_entry_point)
            else:
                raise ValueError(f"Unable to find entry path: {entry} ({env_name})")
        except Exception as e:
            raise ValueError(str(e))
        # parse loaded cfg according to the RL library
        if self.library == "rl_games":
            try:
                self._parse_rl_games(cfg)
                self.valid = True
            except AssertionError as e:
                print(f"  |-- [WARNING] rl_games bad config: {e}")
        elif self.library == "rsl_rl":
            try:
                self._parse_rsl_rl(cfg)
                self.valid = True
            except AssertionError as e:
                print(f"  |-- [WARNING] rsl_rl bad config: {e}")
        elif self.library == "skrl":
            self._parse_skrl(cfg)
            self.valid = True
        return self

    def generate_yaml(self) -> tuple[str, str]:
        content = None
        if self.library == "rl_games":
            # generate file name
            filename = os.path.basename(self.path).replace("rl_games", "")
            filename = filename.replace(f"_{self.algorithm}_", "_")
            filename = filename.replace(f"_{self.algorithm}.", ".")
            filename = filename.replace(f"_cfg_", "_")
            filename = filename.replace(f"_cfg.", ".")
            filename = filename.replace(f".yaml", "")
            filename = f"skrl_{filename}_{self.algorithm}_cfg.yaml".replace("__", "_").replace("__", "_")
            path = os.path.join(os.path.dirname(self.path), filename)
            # open template
            with open(os.path.join(self._templates, f"{self.algorithm}_rl_games_yaml")) as file:
                content = file.read()
        if self.library == "rsl_rl":
            token = ""
            if "-flat-" in self.cfg["metadata"]["task"].lower():
                token = "flat"
            elif "-rough-" in self.cfg["metadata"]["task"].lower():
                token = "rough"
            # generate file name
            filename = os.path.basename(self.path).replace("rsl_rl", "")
            filename = filename.replace(f"_{self.algorithm}_", "_")
            filename = filename.replace(f"_{self.algorithm}.", ".")
            filename = filename.replace(f"_cfg_", "_")
            filename = filename.replace(f"_cfg.", ".")
            filename = filename.replace(f".py", "")
            filename = f"skrl_{token}_{filename}_{self.algorithm}_cfg.yaml".replace("__", "_").replace("__", "_")
            path = os.path.join(os.path.dirname(self.path), filename)
            # open template
            with open(os.path.join(self._templates, f"{self.algorithm}_rsl_rl_yaml")) as file:
                content = file.read()
        if not content:
            raise NotImplementedError(self.library)
        # render template
        template = Template(content, keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True)
        content = template.render(self.cfg)
        # save file
        with open(path, "w") as file:
            file.write(content)
        dirname = os.path.dirname(self.path)[len(os.getcwd()) + 1:]
        return dirname, filename

    def generate_python_script(self) -> None:
        def convert_hidden_activation(activations, framework):
            mapping = {
                "torch": {
                    "": "Identity",
                    "relu": "ReLU",
                    "tanh": "Tanh",
                    "sigmoid": "Sigmoid",
                    "leaky_relu": "LeakyReLU",
                    "elu": "ELU",
                    "softplus": "Softplus",
                    "softsign": "Softsign",
                    "selu": "SELU",
                    "softmax": "Softmax",
                },
                "jax": {
                    "relu": "relu",
                    "tanh": "tanh",
                    "sigmoid": "sigmoid",
                    "leaky_relu": "leaky_relu",
                    "elu": "elu",
                    "softplus": "softplus",
                    "softsign": "soft_sign",
                    "selu": "selu",
                    "softmax": "softmax",
                },
            }
            return [mapping[framework][activation] for activation in activations]

        task_name = "_".join([item.lower() for item in self.cfg["metadata"]["task"].split("-")[1:-1]])
        for framework in ["torch", "jax"]:
            content = ""
            if self.library == "skrl":
                # generate file name
                os.makedirs("skrl_examples", exist_ok=True)
                path = os.path.join("skrl_examples", f"{framework}_{task_name}_ppo.py")
                with open(os.path.join(self._templates, f"ppo_skrl_py_{framework}")) as file:
                    content = file.read()
            if not content:
                raise ValueError
            # update config
            self.cfg["models"]["policy"][f"hidden_activation__{framework}"] = convert_hidden_activation(
                self.cfg["models"]["policy"]["hidden_activation"], framework
            )
            self.cfg["models"]["value"][f"hidden_activation__{framework}"] = convert_hidden_activation(
                self.cfg["models"]["value"]["hidden_activation"], framework
            )
            # render template
            template = Template(content, keep_trailing_newline=True, trim_blocks=True, lstrip_blocks=True)
            content = template.render(self.cfg)
            # save file
            with open(path, "w") as file:
                file.write(content)


if __name__ == "__main__":

    stats = []

    for env_name, env_data in gym.envs.registry.items():
        # ignore non-Isaac Lab envs
        if not env_name.lower().startswith("isaac-"):
            continue
        stats.append({"env": env_name, "registered": {}, "generated": []})
        print(f"\n{'=' * len(env_name)}\n{env_name}\n{'=' * len(env_name)}")

        # get number of environments
        num_envs = load_cfg_from_registry(env_name, "env_cfg_entry_point").scene.num_envs

        # get libraries config
        library = "rl_games"
        rl_games_configs = [
            Config(library).parse(entry, env_name, num_envs)
            for entry, _ in env_data.kwargs.items()
            if entry.startswith(library)
        ]
        if len(rl_games_configs):
            stats[-1]["registered"]["rl_games"] = rl_games_configs

        library = "rsl_rl"
        rsl_rl_configs = [
            Config(library).parse(entry, env_name, num_envs)
            for entry, _ in env_data.kwargs.items()
            if entry.startswith(library)
        ]
        if len(rsl_rl_configs):
            stats[-1]["registered"]["rsl_rl"] = rsl_rl_configs

        library = "skrl"
        skrl_configs = [
            Config(library).parse(entry, env_name, num_envs)
            for entry, _ in env_data.kwargs.items()
            if entry.startswith(library)
        ]
        if len(skrl_configs):
            stats[-1]["registered"]["skrl"] = skrl_configs

        # ignore PLAY configs: Isaac-ENV-NAME-Play-v0
        if env_name.lower().endswith("-play-v0"):
            stats[-1]["generated"].append({"filename": "-"})
            continue

        # generate config file
        generated = False
        # rl_games config
        if len(rl_games_configs):
            assert len(rl_games_configs) == 1
            config = rl_games_configs[0]
            if config.valid:
                dirname, filename = config.generate_yaml()
                stats[-1]["generated"].append({"dirname": dirname, "filename": filename, "library": "rl_games"})
                generated = True
         # rsl_rl config
        if not generated and len(rsl_rl_configs):
            assert len(rsl_rl_configs) == 1
            config = rsl_rl_configs[0]
            if config.valid:
                dirname, filename = config.generate_yaml()
                stats[-1]["generated"].append({"dirname": dirname, "filename": filename, "library": "rsl_rl"})
                generated = True

        # generate Python scripts
        library = "skrl"
        skrl_configs = [
            Config(library).parse(entry, env_name, num_envs)
            for entry, _ in env_data.kwargs.items()
            if entry.startswith(library)
        ]
        if len(skrl_configs):
            assert len(skrl_configs) == 1
            config = skrl_configs[0]
            if config.valid:
                config.generate_python_script()

    print()
    print("#################################")
    print()

    cmd = "git status --porcelain | grep skrl_.*.yaml"
    git_status = subprocess.check_output(cmd, shell=True, text=True).split("\n")

    def status(generated):
        for item in generated:
            if item and "dirname" in item:
                for row in git_status:
                    if os.path.join(item["dirname"], item["filename"]) in row:
                        return row.lstrip().split(" ")[0]
        return ""

    table = PrettyTable()
    table.field_names = ["Task", "Registered", "RL libraries", "Generated", "Status"]
    table.align["Task"] = "l"
    table.align["RL libraries"] = "l"
    table.align["Generated"] = "l"
    for data in stats:
        if "skrl" in data["registered"]:
            registered = f'- {len(data["registered"]["skrl"])} -' if len(data["registered"]["skrl"]) > 1 else ""
            filenames = [item.get("filename") for item in data["generated"]]
            if filenames == ["-"]:
                pass
            elif filenames:
                exist = False
                for item in data["registered"]["skrl"]:
                    if item.filename in filenames:
                        exist = True
                if not exist:
                    registered = f"{registered} other" if registered else "other"
        else:
            registered = "No"
        libraries = sorted(data["registered"].keys())
        filenames = [f'{item.get("filename")}' + (f' ({item.get("library")})' if "library" in item else "")
                     for item in data["generated"] if item]
        table.add_row([
            data["env"],
            registered,
            ", ".join(libraries),
            ", ".join(filenames),
            status(data["generated"]),
        ])
    print(table)
    print()
