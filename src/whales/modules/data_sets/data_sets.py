from whales.modules.data_files.data_files import DataFile
from whales.modules.module import Module


class DataSet(Module):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.datafiles = []
        self._iterations = 1

    def add_data_file(self, data_file: DataFile):
        """Add data file to data set"""
        self.datafiles.append(data_file)

    def remove_data_file(self, data_file: DataFile):
        """Remove data file from data set"""
        self.datafiles.remove(data_file)

