import numpy as np
from src.whales.modules.pre_processing.pre_processing import PreProcessing


class Scale(PreProcessing):
    def __init__(self, logger=None):
        super(Scale, self).__init__(logger)
        self.description = "Center to the mean and scale to unit variance"
        self.needs_fitting = False
        self.parameters = {
        }

    def method_transform(self, data):
        out = np.copy(data)
        out -= out.mean()
        out /= out.std()
        return out


PipelineMethod = Scale
