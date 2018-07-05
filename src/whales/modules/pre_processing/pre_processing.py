import time

from whales.modules.data_files.data_files import DataFile
from whales.modules.module import Module


class PreProcessing(Module):
    def __init__(self, logger=None):
        super().__init__(logger)

    def transform(self):
        if self.needs_fitting and not self.is_fitted:
            raise RuntimeError(f"Pre processing {self} has not been fitted")
        data = self.parameters["data"]
        if issubclass(data.__class__, DataFile):
            self.parameters["data"] = data
        t0 = time.time()
        res = self.method_transform()
        tf = time.time()
        self.logger.debug(f"Pre processing {self} transformation took {tf - t0} s")
        return res

    def method_transform(self):
        raise NotImplementedError("Transformation method has not been implemented")
