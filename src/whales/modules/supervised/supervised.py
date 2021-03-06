import pandas as pd

from whales.modules.data_files.data_files import DataFile
from whales.modules.module import Module


class Supervised(Module):
    def __init__(self, logger=None):
        super().__init__(logger=logger)
        self.parameters = {
        }
        self.private_parameters = {
            "data": [],
            "target": []
        }
        self.type = "supervised"

    def fit(self):
        data = self.all_parameters["data"]
        labeled_data = data.get_labeled_data()
        self.private_parameters["target"] = labeled_data["labels"]
        self.private_parameters["data"] = labeled_data["data"]
        self.method_fit()
        self.is_fitted = True

        # Clean for saving model
        self.private_parameters["data"] = None
        self.private_parameters["target"] = None

    def method_fit(self):
        raise NotImplementedError

    def method_predict(self):
        raise NotImplementedError

    def predict(self):
        data = self.all_parameters["data"]
        if issubclass(data.__class__, DataFile):
            self.private_parameters["data"] = data.data.values
        else:
            raise ValueError("Data input should be a Data File")
        res = self.method_predict()
        inds = data.data.index
        return pd.Series(res, index=inds)
