import numpy as np
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Min(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Minimum"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self, data):
        return np.min(data, axis=1).reshape(-1, 1)


PipelineMethod = Min
