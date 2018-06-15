import numpy as np
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Range(FeatureExtraction):
    description = "Range"

    def __init__(self, logger=None):
        super(Range, self).__init__(logger)
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self, data):
        min_value = np.min(data, axis=1)
        max_value = np.max(data, axis=1)
        range_value = max_value - min_value
        return range_value.reshape(-1, 1)


PipelineMethod = Range
