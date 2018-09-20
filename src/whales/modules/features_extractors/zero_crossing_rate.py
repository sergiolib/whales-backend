import numpy as np
import pandas as pd
from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class ZeroCrossingRate(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Zero crossing rate"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        data_file = self.all_parameters["data_file"]
        data = data_file.data.values
        signs = np.sign(data)
        sign_change = np.array(signs[:, 1:] != signs[:, :-1]).astype(int)
        res = np.nansum(sign_change, axis=1) / data.shape[1].reshape(1, -1)
        data = pd.DataFrame(res, index=data_file.data.index, columns=["zero_crossing_rate"])
        fdf = FeatureDataFile()
        fdf._data = data
        return fdf


PipelineMethod = ZeroCrossingRate
