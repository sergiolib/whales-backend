from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Energy(FeatureExtraction):
    def __init__(self, logger=None):
        super(Energy, self).__init__(logger)
        self.needs_fitting = False
        self.description = "Energy"
        self.parameters = {

        }

    def method_transform(self, data):
        return (data * data).sum(axis=1)


PipelineMethod = Energy
