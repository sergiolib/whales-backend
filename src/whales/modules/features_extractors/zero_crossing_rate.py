import numpy as np
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class ZeroCrossingRate(FeatureExtraction):
    description = "Zero crossing rate"
    parameters = {}

    def __init__(self, logger=None):
        super(ZeroCrossingRate, self).__init__(logger)
        self.needs_fitting = False

    def method_transform(self, data):
        signs = np.sign(data)
        sign_change = (signs[:, 1:] != signs[:, :-1]).astype(int)
        res = np.sum(sign_change, axis=1) / data.shape[1]
        return res.reshape(-1, 1)


PipelineMethod = ZeroCrossingRate
