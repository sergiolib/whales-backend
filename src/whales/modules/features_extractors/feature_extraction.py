import time
import pandas as pd

from whales.modules.data_files.data_files import DataFile
from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.module import Module


class FeatureExtraction(Module):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.parameters = {
            "data": []
        }

    def fit(self):
        if self.needs_fitting is False:
            return
        data = self.parameters["data"]
        if issubclass(data.__class__, DataFile):
            self.parameters["data"] = data.data.values
        if data.ndim == 1:
            self.parameters["data"] = data.reshape(-1, 1)
        t0 = time.time()
        self.method_fit()
        tf = time.time()
        self.logger.debug(f"Features extractor {self} fitting took {tf - t0} s")

    def method_fit(self):
        raise NotImplementedError(f"Fitting method in {self} features extractor has not been implemented")

    def transform(self):
        if self.needs_fitting and not self.is_fitted:
            raise RuntimeError(f"Features extractor {self} has not been fitted")
        data = self.parameters["data"]
        if issubclass(data.__class__, DataFile):
            self.parameters["data"] = data.data.values
        if self.parameters["data"].ndim == 1:
            self.parameters["data"] = self.parameters["data"].reshape(-1, 1)
        t0 = time.time()
        out = self.method_transform()
        res = FeatureDataFile()
        res.data = pd.DataFrame(out, columns=[f"{self.short_name.lower()}_{i}" for i in range(out.shape[1])])
        tf = time.time()
        self.logger.debug(f"Features extractor {self} transformation took {tf - t0} s")
        return res

    def method_transform(self):
        raise NotImplementedError("Transformation method has not been implemented")
