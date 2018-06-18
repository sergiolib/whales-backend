from whales.modules.datafiles.datafiles import Datafile
from whales.modules.datasets.datasets import DataSet


class FilesFoldDataSet(DataSet):
    """Perform K fold where every file is treated as a separate source"""
    def __init__(self, logger=None):
        super(FilesFoldDataSet, self).__init__(logger)
        self.current_file = 0

        self.parameters = {
            "training": True,
            "testing": True,
            "validation": True
        }

    def get_training(self):
        if not self.parameters["training"]:
            return None
        for i in list(range(len(self.datafiles) - 1)) + [-1]:
            curr_datafiles = self.datafiles.copy()
            if self.parameters["validation"]:
                curr_datafiles.pop(i + 1)  # i + 1 is validation
            if self.parameters["testing"]:
                curr_datafiles.pop(i)  # i is testing
            yield Datafile().concatenate(curr_datafiles)

    def get_testing(self):
        if not self.parameters["testing"]:
            return None
        for i in range(len(self.datafiles)):
            curr_datafiles = self.datafiles.copy()
            testing = curr_datafiles.pop(i)
            yield Datafile().concatenate([testing])

    def get_validation(self):
        if not self.parameters["validation"]:
            return None
        for i in list(range(len(self.datafiles) - 1)) + [-1]:
            curr_datafiles = self.datafiles.copy()
            validation = curr_datafiles.pop(i + 1)
            yield Datafile().concatenate([validation])


PipelineDataSet = FilesFoldDataSet
