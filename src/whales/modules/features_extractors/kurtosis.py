import scipy.stats
import pandas as pd

from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Kurtosis(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Kurtosis"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        data_file = self.all_parameters["data_file"]
        data = data_file.data.values
        res = scipy.stats.kurtosis(data, axis=1, nan_policy="omit").reshape(-1, 1)
        f = FeatureDataFile()
        f.data = pd.DataFrame(res, columns=["kurtosis"], index=data_file.data.index)
        return f


PipelineMethod = Kurtosis
