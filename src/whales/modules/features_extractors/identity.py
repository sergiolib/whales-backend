from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Identity(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Identity"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        data = self.all_parameters["data"]
        # Caution with nan values as they cannot go into the classifiers
        return data


PipelineMethod = Identity
