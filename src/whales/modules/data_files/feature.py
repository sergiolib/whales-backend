import pandas as pd

from whales.modules.data_files.data_files import DataFile


class FeatureDataFile(DataFile):
    def __init__(self, data_file=None, logger=None):
        super().__init__(logger=logger)
        if data_file is not None:
            self._data = data_file.data
            self.metadata = data_file.metadata

    def concatenate(self, datafiles_list):
        new_df = self.__class__()
        data = [df.data for df in datafiles_list]
        data = pd.concat(data, axis=1)
        new_df.data = data
        return new_df


class AudioSegments(FeatureDataFile):
    def __init__(self, data_file=None, logger=None):
        super().__init__(data_file=data_file, logger=logger)

        self.parameters = {
            "labels": None,
        }

    def __repr__(self):
        if hasattr(self, "data") and self.data is not None:
            n_audios = len(self.data)
            return " ".join([self.__class__.__name__, f"({n_audios} segments)"])
        return self.__class__.__name__

    def add_segment(self, data, label):
        labels = self.parameters["labels"]
        if labels is None:
            self.data = pd.DataFrame(data).T
            labels = pd.Series({data.name: label})
        else:
            self.data = self.data.append(data, sort=False)
            labels = labels.append(pd.Series({data.name: label}))
        self.parameters["labels"] = labels
