class BaseWorkflowDataset:
    def __init__(self, config):
        self.config = config

    def load(self):
        raise NotImplementedError
