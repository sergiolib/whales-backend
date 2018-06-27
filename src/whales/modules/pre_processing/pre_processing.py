import time

from whales.modules.module import Module


class PreProcessing(Module):
    def __init__(self, logger=None):
        super().__init__(logger)

    def transform(self, data_file):
        if self.needs_fitting and not self.is_fitted:
            raise RuntimeError(f"Pre processing {self} has not been fitted")
        t0 = time.time()
        res = self.method_transform(data_file)
        self.logger.debug(f"Transformed data with pre processing {self}")
        tf = time.time()
        self.logger.debug(f"Pre processing {self} fitting took {tf - t0} s")
        return res

    def method_transform(self, data_file):
        raise NotImplementedError("Transformation method has not been implemented")
