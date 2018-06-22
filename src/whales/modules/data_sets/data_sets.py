from whales.modules.data_files.data_files import DataFile
from whales.modules.module import Module


class DataSet(Module):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.datafiles = []

    def add_datafile(self, datafile: DataFile):
        """Add data file to data set"""
        self.datafiles.append(datafile)

    def remove_datafile(self, datafile: DataFile):
        """Remove data file from data set"""
        self.datafiles.remove(datafile)

