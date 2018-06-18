from whales.modules.module import Module


class Supervised(Module):
    def fit(self, data, target):
        self.method_fit(data, target)

    def method_fit(self, data, target):
        raise NotImplementedError

    def method_predict(self, data):
        raise NotImplementedError

    def predict(self, data):
        return self.method_predict(data)
