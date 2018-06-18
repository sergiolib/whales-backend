from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Demo(FeatureExtraction):
    def __init__(self, logger=None):  # There should be no parameters here
        super(Demo, self).__init__(logger)

        self.description = "This is a demo"

        # Class attributes go here
        # Eg.:
        # self._model = MFCC()

        self.parameters = {
            # Add parameters here
        }

    def method_fit(self, data, **kwargs):
        # Here goes the training, if needed. Features always use data, and sometimes other inputs
        pass

    def method_transform(self, data, **kwargs):
        # Here goes the training. Same parameters as in fit
        return None


# PipelineMethod = Demo  # This line is mandatory to make the method loadable by the pipeline
