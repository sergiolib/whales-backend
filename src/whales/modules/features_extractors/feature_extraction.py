import time
import pandas as pd
import numpy as np

from whales.modules.data_files.audio import AudioDataFile
from whales.modules.data_files.data_files import DataFile
from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.module import Module


class FeatureExtraction(Module):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.private_parameters = {}
        self.parameters = {}
        self.metadata = {}

    def fit(self):
        if self.needs_fitting:
            data = self.all_parameters["data_file"]
            if not issubclass(data.__clas__, DataFile):
                raise AttributeError("Data parameter should be a proper data file")
            if issubclass(data.__class__, DataFile):
                self.private_parameters["data_file"] = data.data.values
            if data.ndim == 1:
                self.private_parameters["data_file"] = data.reshape(-1, 1)
            t0 = time.time()
            self.method_fit()
            tf = time.time()
            self.logger.debug(f"Features extractor {self} fitting took {tf - t0} s")
        self.private_parameters["data_file"] = None

    def method_fit(self):
        raise NotImplementedError(f"Fitting method in {self} features extractor has not been implemented")

    def transform(self):
        if self.needs_fitting and not self.is_fitted:
            raise RuntimeError(f"Features extractor {self} has not been fitted")
        t0 = time.time()
        df = self.private_parameters["data_file"]
        res = []
        for st, en in df.metadata["starts_stops"]:
            local_df = AudioDataFile(df)
            local_df._data = local_df._data[st:en]
            self.private_parameters["data_file"] = local_df
            res.append(self.method_transform())
        res = FeatureDataFile(res[0].metadata["feature_name"]).concatenate(res, axis=0)
        res.label_name = self.all_parameters["data_file"].label_name
        tf = time.time()
        self.logger.debug(f"Features extractor {self} transformation took {tf - t0} s")
        return res

    def method_transform(self):
        raise NotImplementedError("Transformation method has not been implemented")
