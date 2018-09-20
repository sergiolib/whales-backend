import numpy as np
import pandas as pd

from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Min(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Minimum"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        data_file = self.all_parameters["data_file"]
        data = data_file.data.values
        res = np.nanmin(data, axis=1).reshape(-1, 1)
        data = pd.DataFrame(res, index=data_file.data.index, columns=["min"])
        fdf = FeatureDataFile()
        fdf._data = data
        return fdf


PipelineMethod = Min
