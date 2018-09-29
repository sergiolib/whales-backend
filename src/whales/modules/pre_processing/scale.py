from whales.modules.pre_processing.pre_processing import PreProcessing
from whales.modules.data_files.audio import AudioDataFile


class Scale(PreProcessing):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Center to the mean and scale to unit variance"
        self.parameters = {}

    def method_transform(self):
        data_file = self.all_parameters["data_file"]
        if issubclass(data_file.__class__, AudioDataFile):
            for st, en in data_file.metadata["starts_stops"]:
                data_file.data[st:en] -= data_file.data[st:en].mean()
                data_file.data[st:en] /= data_file.data[st:en].std()
            return data_file
        else:
            raise NotImplementedError


PipelineMethod = Scale
