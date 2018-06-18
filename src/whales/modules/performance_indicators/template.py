from whales.modules.performance_indicators.performance_indicator import PerformanceIndicator


class Demo(PerformanceIndicator):
    def __init__(self, logger=None):  # There should be no parameters here
        super(Demo, self).__init__(logger)

        self.description = "This is a demo"

        # Class attributes go here
        # Eg.:
        # self._model = SKLearn.Recall()

        self.parameters = {
            # Add parameters here. "Inputs" have to be placed here too
        }

    def method_fit(self):
        # Here goes the training, if needed. Performance indicators usually don't get trained
        pass

    def method_evaluate(self):
        # Here goes the training. Inputs are stated inside the parameters
        # This is like that because performance indicators, and specially visualizations sometimes use complex inputs
        # that are hard to generalize.
        return None


# PipelineMethod = Demo  # This line is mandatory to make the method loadable by the pipeline
