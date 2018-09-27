import pandas as pd
import numpy as np
from whales.modules.data_files.data_files import DataFile
from whales.modules.module import Module


class Unsupervised(Module):
    def __init__(self, logger=None):
        super().__init__(logger=logger)
        self.parameters = {}
        self.private_parameters = {
            "data": [],
            "target": []
        }
        self.type = "unsupervised"

    def fit(self):
        if self.needs_fitting is True:
            df = self.all_parameters["data"]
            self.private_parameters["data"] = df
            self.method_fit()
            self.is_fitted = True
        self.private_parameters["data"] = None

    def method_fit(self):
        raise NotImplementedError

    def method_predict(self):
        raise NotImplementedError

    def predict(self):
        df = self.all_parameters["data"]
        self.private_parameters["data"] = df
        res = self.method_predict().astype(int)
        res = res - res.min()
        return pd.Series(res, index=df.data.index)
