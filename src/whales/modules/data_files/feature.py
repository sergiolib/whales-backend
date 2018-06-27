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
