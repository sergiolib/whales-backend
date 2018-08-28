import numpy as np
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class ZeroCrossingRate(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Zero crossing rate"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        data = self.all_parameters["data"]
        signs = np.sign(data)
        sign_change = np.array(signs[:, 1:] != signs[:, :-1]).astype(int)
        res = np.nansum(sign_change, axis=1) / data.shape[1]
        return res.reshape(1, -1)


PipelineMethod = ZeroCrossingRate
