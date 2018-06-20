from whales.modules.datafiles.datafiles import Datafile
from whales.modules.module import Module


class DataSet(Module):
    def __init__(self, logger=None):
        super(DataSet, self).__init__(logger)
        self.datafiles = []

    def add_datafile(self, datafile: Datafile):
        """Add data file to data set"""
        self.datafiles.append(datafile)

    def remove_datafile(self, datafile: Datafile):
        """Remove data file from data set"""
        self.datafiles.remove(datafile)

