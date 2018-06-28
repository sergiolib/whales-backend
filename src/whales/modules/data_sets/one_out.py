from whales.modules.data_files.data_files import DataFile
from whales.modules.data_files.feature import AudioSegments
from whales.modules.data_sets.data_sets import DataSet


class OneDataFileOut(DataSet):
    """Perform K fold where every file is treated as a separate source"""
    def __init__(self, logger=None):
        super().__init__(logger)
        self.current_file = 0

        self.description = """K Folding of three separate sets for every fold"""

        self.parameters = {
            "training": True,
            "testing": True,
            "validation": True
        }

    @property
    def iterations(self):
        return len(self.datafiles)

    def get_training(self):
        if not self.parameters["training"]:
            return None
        for i in list(range(len(self.datafiles) - 1)) + [-1]:
            curr_datafiles = self.datafiles.copy()
            if self.parameters["validation"]:
                curr_datafiles.pop(i + 1)  # i + 1 is validation
            if self.parameters["testing"]:
                curr_datafiles.pop(i)  # i is testing
            yield AudioSegments().concatenate(curr_datafiles)

    def get_testing(self):
        if not self.parameters["testing"]:
            return None
        for i in range(len(self.datafiles)):
            curr_datafiles = self.datafiles.copy()
            testing = curr_datafiles.pop(i)
            yield AudioSegments().__class__().concatenate([testing])

    def get_validation(self):
        if not self.parameters["validation"]:
            return None
        for i in list(range(len(self.datafiles) - 1)) + [-1]:
            curr_datafiles = self.datafiles.copy()
            validation = curr_datafiles.pop(i + 1)
            yield AudioSegments().__class__().concatenate([validation])

    def get_data_sets(self):
        if self.parameters["training"] and self.parameters["testing"] and self.parameters["validation"]:
            for tr, te, val in self.get_training(), self.get_testing(), self.get_validation():
                yield tr, te, val
        elif self.parameters["training"] and self.parameters["testing"]:
            for tr, te in self.get_training(), self.get_testing():
                yield tr, te


PipelineDataSet = OneDataFileOut
