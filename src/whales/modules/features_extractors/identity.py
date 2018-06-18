from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Identity(FeatureExtraction):
    def __init__(self, logger=None):
        super(Identity, self).__init__(logger)
        self.description = "Identity"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self, data):
        return data


PipelineMethod = Identity
