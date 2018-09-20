import pandas as pd
from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Identity(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Identity"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        new_data_file = FeatureDataFile()
        data_file = self.all_parameters["data_file"]
        # Caution with nan values as they cannot go into the classifiers
        # Removing it in the mean time...

        data = data_file.data.dropna()
        data.columns = [f"identity_{i}" for i, _ in enumerate(data.columns)]
        new_data_file._data = data
        return new_data_file


PipelineMethod = Identity
