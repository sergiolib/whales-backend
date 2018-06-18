from whales.modules.performance_indicators.performance_indicator import PerformanceIndicator


class Demo(PerformanceIndicator):
    def __init__(self, logger=None):  # There should be no parameters here
        super(Demo, self).__init__(logger)

        self.description = "This is a demo"

        # Class attributes go here
        # Eg.:
        # self._model = SKLearn.Recall()

        self.parameters = {
            # Add parameters here
        }

    def method_fit(self, **kwargs):
        # Here goes the training, if needed. Performance indicators usually don't get trained
        pass

    def method_evaluate(self, **kwargs):
        # Here goes the training. Same parameters as in fit
        return None


# PipelineMethod = Demo  # This line is mandatory to make the method loadable by the pipeline
