
class Environment:
    def __init__(self):
        self.num_envs = 0

        self.states_buffer = None
        self.rewards_buffer = None
        self.dones_buffer = None
