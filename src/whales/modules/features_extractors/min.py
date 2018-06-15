import numpy as np
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Min(FeatureExtraction):
    description = "Minimum"

    def __init__(self, logger=None):
        super(Min, self).__init__(logger)
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self, data):
        return np.min(data, axis=1).reshape(-1, 1)


PipelineMethod = Min
