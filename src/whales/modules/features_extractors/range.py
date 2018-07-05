import numpy as np
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Range(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Range"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        data = self.parameters["data"]
        min_value = np.nanmin(data, axis=1)
        max_value = np.nanmax(data, axis=1)
        range_value = max_value - min_value
        return range_value.reshape(-1, 1)


PipelineMethod = Range
