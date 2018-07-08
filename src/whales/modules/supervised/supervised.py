import pandas as pd

from whales.modules.data_files.data_files import DataFile
from whales.modules.module import Module


class Supervised(Module):
    def __init__(self, logger=None):
        super().__init__(logger=logger)
        self.parameters = {
            "data": [],
            "target": []
        }

    def fit(self):
        data = self.parameters["data"]
        if issubclass(data.__class__, DataFile):
            inds = data.data.index
            self.parameters["data"] = data.data.loc[inds].values
            self.parameters["target"] = data.metadata["labels"].loc[inds].values
        self.method_fit()
        self.is_fitted = True

        # Clean for saving model
        self.parameters["data"] = None
        self.parameters["target"] = None

    def method_fit(self):
        raise NotImplementedError

    def method_predict(self):
        raise NotImplementedError

    def predict(self):
        data = self.parameters["data"]
        if issubclass(data.__class__, DataFile):
            inds = data.data.index
            self.parameters["data"] = data.data.loc[inds].values
        else:
            raise ValueError("Data input should be a Data File")
        res = self.method_predict()
        return pd.Series(res, index=data.data.index)
