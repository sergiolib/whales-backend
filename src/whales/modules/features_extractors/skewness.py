import scipy.stats
import pandas as pd
from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Skewness(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Skewness"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        data_file = self.all_parameters["data_file"]
        data = data_file.data.values
        res = scipy.stats.skew(data, axis=1, nan_policy="omit").reshape(-1, 1)
        data = pd.DataFrame(res, index=data_file.data.index, columns=["skewness"])
        fdf = FeatureDataFile()
        fdf._data = data
        return fdf


PipelineMethod = Skewness
