import numpy as np
from src.whales.modules.pre_processing.pre_processing import PreProcessing
from whales.modules.data_files.audio_windows import AudioWindowsDataFile


class Scale(PreProcessing):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Center to the mean and scale to unit variance"
        self.parameters = {}

    def method_transform(self, data_file):
        if type(data_file) is AudioWindowsDataFile:
            out = data_file.data["data_0"]
            out -= out.mean()
            out /= out.std()
            return data_file
        else:
            raise NotImplementedError


PipelineMethod = Scale
