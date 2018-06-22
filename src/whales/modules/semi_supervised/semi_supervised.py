from whales.modules.module import Module


class SemiSupervised(Module):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.parameters = {
            "unlabeled_placeholder": -1,
        }

    def fit(self, data, target):
        self.method_fit(data, target)

    def method_fit(self, data, target):
        raise NotImplementedError

    def method_predict(self, data, target):
        raise NotImplementedError

    def predict(self, data, target):
        return self.method_predict(data, target)
