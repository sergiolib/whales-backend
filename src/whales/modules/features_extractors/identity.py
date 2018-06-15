from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Identity(FeatureExtraction):
    description = "Identity"
    parameters = {}

    def __init__(self, logger=None):
        super(Identity, self).__init__(logger)
        self.needs_fitting = False

    def method_transform(self, data):
        return data


PipelineMethod = Identity
