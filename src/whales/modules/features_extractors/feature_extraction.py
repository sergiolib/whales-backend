import time
import pandas as pd
import numpy as np

from whales.modules.data_files.data_files import DataFile
from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.module import Module


class FeatureExtraction(Module):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.private_parameters = {
            "data": []
        }
        self.parameters = {}

    def fit(self):
        if self.needs_fitting:
            data = self.all_parameters["data"]
            if not issubclass(data.__clas__, DataFile):
                raise AttributeError("Data parameter should be a proper data file")
            if issubclass(data.__class__, DataFile):
                self.private_parameters["data"] = data.data.values
            if data.ndim == 1:
                self.private_parameters["data"] = data.reshape(-1, 1)
            t0 = time.time()
            self.method_fit()
            tf = time.time()
            self.logger.debug(f"Features extractor {self} fitting took {tf - t0} s")
        self.private_parameters["data"] = None

    def method_fit(self):
        raise NotImplementedError(f"Fitting method in {self} features extractor has not been implemented")

    def transform(self):
        if self.needs_fitting and not self.is_fitted:
            raise RuntimeError(f"Features extractor {self} has not been fitted")
        data = self.all_parameters["data"]
        if issubclass(data.__class__, DataFile):
            self.private_parameters["data"] = data.data.values
        if self.all_parameters["data"].ndim == 1:
            self.private_parameters["data"] = self.all_parameters["data"].reshape(1, -1)
        t0 = time.time()
        out = self.method_transform()
        if np.isnan(out).any():
            raise RuntimeError(f"Feature {self} returned a NaN")
        res = FeatureDataFile()
        res.data = pd.DataFrame(out, columns=[f"{self.short_name.lower()}_{i}" for i in range(out.shape[1])])
        res.label_name = data.label_name
        tf = time.time()
        self.logger.debug(f"Features extractor {self} transformation took {tf - t0} s")
        return res

    def method_transform(self):
        raise NotImplementedError("Transformation method has not been implemented")
