import numpy as np
from src.whales.modules.preprocessing.preprocessing import Preprocessing


class Scale(Preprocessing):
    def __init__(self, logger=None):
        super(Scale, self).__init__(logger)
        self.description = """Center to the meand and scale to unit variance"""
        self.needs_fitting = False
        self.parameters = {
            # TODO: add relevant parameters
        }

    def method_transform(self, data):
        out = np.copy(data)
        out -= out.mean()
        out /= out.std()
        return out


PipelineMethod = Scale
