
class Agent:
    def __init__(self, env, networks, memory=None, cfg: dict = {}) -> None:
        self.device = "cuda:0"

        self.env = env
        self.networks = networks
        self.memory = memory
        self.cfg = cfg

    def act(self, states, inference=False):
        raise NotImplementedError

    def pre_rollout(self):
        pass

    def inter_rollout(self, iteration, rollout):
        pass

    def post_rollout(self):
        pass
    