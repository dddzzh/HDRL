class BaseScheduler:
    def __init__(self, params=None, seed=42):
        self.params = params or {}
        self.seed = seed

    def reset(self):
        pass

    def select_action(self, state):
        raise NotImplementedError
