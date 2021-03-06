import pandas as pd

from whales.modules.data_files.data_files import DataFile
from whales.modules.module import Module


class SemiSupervised(Module):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.parameters = {}
        self.private_parameters = {
            "unlabeled_placeholder": -1,
            "data": [],
            "target": []
        }
        self.type = "semi_supervised"

    def fit(self):
        data = self.all_parameters["data"]
        if issubclass(data.__class__, DataFile):
            labeled_data = data.get_labeled_data()
            self.private_parameters["target"] = labeled_data["labels"]
            self.private_parameters["data"] = labeled_data[labeled_data.columns.drop("labels")]
        self.method_fit()

    def method_fit(self):
        raise NotImplementedError

    def method_predict(self):
        raise NotImplementedError

    def predict(self):
        data = self.all_parameters["data"]
        if issubclass(data.__class__, DataFile):
            inds = data.data.index
            self.private_parameters["data"] = data.data.loc[inds].values
            self.private_parameters["target"] = data.metadata["labels"].loc[inds].values
        else:
            raise ValueError("Data input should be a Data File")
        res = self.method_predict()
        return pd.Series(res, index=data.data.index)
