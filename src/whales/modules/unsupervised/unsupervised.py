from whales.modules.module import Module


class Unsupervised(Module):
    def fit(self, data):
        self.method_fit(data)

    def method_fit(self, data):
        raise NotImplementedError

    def method_predict(self, data):
        raise NotImplementedError

    def predict(self, data):
        return self.method_predict(data)