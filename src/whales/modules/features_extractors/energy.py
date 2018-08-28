import numpy as np
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Energy(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Energy"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        data = self.all_parameters["data"]
        return np.nansum(data * data, axis=1).reshape(-1, 1)


PipelineMethod = Energy
