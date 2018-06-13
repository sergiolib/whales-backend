import numpy as np
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Min(FeatureExtraction):
    def __init__(self, logger=None):
        super(Min, self).__init__(logger)
        self.needs_fitting = False
        self.description = "Minimum"
        self.parameters = {

        }

    def method_transform(self, data):
        return np.min(data, axis=1).reshape(-1, 1)


PipelineMethod = Min