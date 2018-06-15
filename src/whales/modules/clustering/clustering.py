from whales.modules.module import Module


class Clustering(Module):
    def fit(self, data):
        # Do not re implement this method, instead implement method_
        self.method_fit(data)

    def method_fit(self, data):
        raise NotImplementedError

    def method_predict(self, data):
        raise NotImplementedError

    def predict(self, data):
        return self.method_predict(data)
