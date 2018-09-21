import pandas as pd

from whales.modules.data_files.data_files import DataFile


class FeatureDataFile(DataFile):
    def __init__(self, feature_name, data_file=None, logger=None):
        super().__init__(logger=logger)
        self.metadata["labels"] = None
        self.label_name = {}
        self.metadata["feature_name"] = feature_name

        if data_file is not None:
            self._data = data_file.data
            self.metadata = data_file.metadata

    @property
    def name_label(self):
        return {b: a for a, b in self.label_name.items()}

    def concatenate(self, datafiles_list):
        new_df = self.__class__(self.metadata["feature_name"])

        data = [df.data for df in datafiles_list]
        label_name = [df.label_name for df in datafiles_list]
        for l in label_name:
            new_df.label_name.update(l)
        ind = data[0].index
        data = [d.loc[ind] for d in data]
        data = pd.concat(data, axis=1)
        new_df._data = data
        return new_df

    def get_labeled_data(self):
        inds = self.data.index
        labels = pd.Series([False for i in range(len(inds))], index=inds)
        for st, en , l in self.metadata["labels"]:
            curr_inds = (st <= inds) & (inds <= en)
            labels[curr_inds] = True

        return pd.concat({"data": self.data.loc[inds], "labels": labels}, axis=1)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        shape = new_data.shape
        feature_name = self.metadata["feature_name"]
        self._data = pd.DataFrame(new_data, columns=[feature_name + f"_{i}" for i in range(shape[1])])
