import time
import pandas as pd

from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.module import Module


class FeatureExtraction(Module):
    def fit(self, data):
        if self.needs_fitting is False:
            return
        t0 = time.time()
        self.method_fit(data)
        tf = time.time()
        self.logger.debug(f"Features extractor {self} fitting took {tf - t0} s")

    def method_fit(self, data):
        raise NotImplementedError(f"Fitting method in {self} features extractor has not been implemented")

    def transform(self, data):
        if self.needs_fitting and not self.is_fitted:
            raise RuntimeError(f"Features extractor {self} has not been fitted")
        t0 = time.time()
        out = self.method_transform(data)
        res = FeatureDataFile()
        res.data = pd.DataFrame(out, columns=[f"{self.short_name.lower()}_{i}" for i in range(out.shape[1])])
        self.logger.debug(f"Transformed data with features extractor {self}")
        tf = time.time()
        self.logger.debug(f"Features extractor {self} transformation took {tf - t0} s")
        return res

    def method_transform(self, data):
        raise NotImplementedError("Transformation method has not been implemented")
