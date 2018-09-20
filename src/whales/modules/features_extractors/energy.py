import numpy as np
import pandas as pd
from whales.modules.features_extractors.feature_extraction import FeatureExtraction
from whales.modules.data_files.feature import FeatureDataFile


class Energy(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Energy"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        data_file = self.all_parameters["data_file"]
        data = data_file.data.values
        res = np.nansum(data ** 2, axis=1).reshape(-1, 1)
        f = FeatureDataFile()
        f.data = pd.DataFrame(res, columns=["energy"], index=data_file.data.index)
        return f


PipelineMethod = Energy
