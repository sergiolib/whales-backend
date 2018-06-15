import time

from src.whales.modules.module import Module


class Preprocessing(Module):
    def __init__(self, logger=None):
        super(Preprocessing, self).__init__(logger)
        self.needs_fitting = False  # Default
        self.is_fitted = False  # Default

    def fit(self, **kwargs):
        if self.needs_fitting is False:
            return
        t0 = time.time()
        self.method_fit(**kwargs)
        tf = time.time()
        self.logger.debug(f"Feature {self} fitting took {tf - t0} s")

    def method_fit(self, **kwargs):
        raise NotImplementedError(f"Fitting method in {self} preprocessing has not been implemented")

    def transform(self, **kwargs):
        if self.needs_fitting and not self.is_fitted:
            raise RuntimeError(f"Preprocessing {self} has not been fitted")
        t0 = time.time()
        res = self.method_transform(**kwargs)
        self.logger.debug(f"Transformed data with preprocessing {self}")
        tf = time.time()
        self.logger.debug(f"Preprocessing {self} fitting took {tf - t0} s")
        return res

    def method_transform(self, **kwargs):
        raise NotImplementedError("Transformation method has not been implemented")
