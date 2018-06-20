import time
from whales.modules.module import Module


class FeatureExtraction(Module):
    def fit(self, data):
        if self.needs_fitting is False:
            return
        t0 = time.time()
        self.method_fit(data)
        tf = time.time()
        self.logger.debug(f"Feature {self} fitting took {tf - t0} s")

    def method_fit(self, data):
        raise NotImplementedError(f"Fitting method in {self} feature has not been implemented")

    def transform(self, data):
        if self.needs_fitting and not self.is_fitted:
            raise RuntimeError(f"Feature {self} has not been fitted")
        t0 = time.time()
        res = self.method_transform(data)
        self.logger.debug(f"Transformed data with feature {self}")
        tf = time.time()
        self.logger.debug(f"Feature {self} fitting took {tf - t0} s")
        return res

    def method_transform(self, data):
        raise NotImplementedError("Transformation method has not been implemented")
