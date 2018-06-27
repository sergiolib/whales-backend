from whales.modules.pre_processing.pre_processing import PreProcessing
from whales.modules.data_files.audio import AudioDataFile


class Scale(PreProcessing):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Center to the mean and scale to unit variance"
        self.parameters = {}

    def method_transform(self, data_file):
        if issubclass(data_file.__class__, AudioDataFile):
            out = data_file.data["data_0"]
            out -= out.mean()
            out /= out.std()
            return data_file
        else:
            raise NotImplementedError


PipelineMethod = Scale
