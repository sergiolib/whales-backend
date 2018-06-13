import scipy.stats

from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Kurtosis(FeatureExtraction):
    def __init__(self, logger=None):
        super(Kurtosis, self).__init__(logger)
        self.needs_fitting = False
        self.description = "Kurtosis"
        self.parameters = {

        }

    def method_transform(self, data):
        return scipy.stats.kurtosis(data, axis=1).reshape(-1, 1)


PipelineMethod = Kurtosis
