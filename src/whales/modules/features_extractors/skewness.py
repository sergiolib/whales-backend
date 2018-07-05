import scipy.stats

from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Skewness(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Skewness"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        data = self.parameters["data"]
        return scipy.stats.skew(data, axis=1, nan_policy="omit").reshape(-1, 1)


PipelineMethod = Skewness
