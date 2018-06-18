import pandas as pd
from copy import deepcopy

from whales.modules.module import Module


class Datafile(Module):
    def __init__(self, datafile=None, logger=None):
        super(Datafile, self).__init__(logger)
        self.metadata = {}
        self._data = None
        self.file_name = None
        self.formatter = None

        if datafile is not None:
            self.metadata = deepcopy(datafile.metadata)
            self.formatter = deepcopy(datafile.formatter)
            self.file_name = deepcopy(datafile.file_name)
            self._data = deepcopy(datafile._data)

    def load_data(self, file_name: str, formatter):
        """Do not actually load the data. Instead, save the access information."""
        self.file_name = file_name
        self.formatter = formatter
        self.metadata = {self.data.name: formatter.read_metadata(file_name)}
        return self

    def save_data(self, file_name: str, formatter):
        """Save the data in self.data into specified file_name with specified formatter and also write the metadata.
        If no data has changed, read data from the Datafile and write it in the file."""
        formatter.write(file_name, self.data)
        formatter.write_metadata(file_name, self.metadata)

    @staticmethod
    def concatenate(datafiles_list):
        """Add datafiles from datafiles_list to new datafile and return it"""
        data = []
        new_df = __class__()
        metadata = {}
        for df in datafiles_list:
            metadata[df.file_name] = df.metadata
            data.append(df.data)

        new_df.data = pd.concat(data, axis=1)
        new_df.metadata = metadata
        return new_df

    @property
    def data(self):
        if self._data is None:
            res = self.formatter.read(self.file_name)
        else:
            res = self._data
        return res

    @data.setter
    def data(self, data):
        self._data = data
