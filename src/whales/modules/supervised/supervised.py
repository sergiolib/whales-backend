from whales.modules.module import Module


class Supervised(Module):
    def fit(self, data):
        self.method_fit(data)

    def method_fit(self, data, targets):
        raise NotImplementedError

    def method_predict(self, data):
        raise NotImplementedError

    def predict(self, data):
        return self.method_predict(data)
