from whales.modules.performance_indicators.performance_indicator import PerformanceIndicator


class Demo(PerformanceIndicator):
    def __init__(self, logger=None):  # There should be no parameters here other than logger
        super().__init__(logger)  # Always call super().__init__(logger) like this

        self.needs_fitting = False  # If True, the method can't be used without fitting it
        # Also, if set, self.method_fit() is mandatory

        self.description = "This is a template"

        # Class attributes go here
        # Eg.:
        # self._model = SKLearn.Recall()

        self.parameters = {
            # prediction = pd.Series(),
            # target = pd.Series(),
            # Add parameters and default values. Also be sure to pass training set as a parameter
        }

    def method_fit(self):
        data = self.parameters["data"]  # Be sure to include the data in the parameters so you pass it to training
        # Here goes the training, if needed. Performance indicators usually don't get trained
        pass

    def method_compute(self):
        data = self.parameters["data"]  # Be sure to include the data in the parameters so you pass it to training.
        # Here goes the method computation.
        return None


# PipelineMethod = Demo  # This line is mandatory to make the method loadable by the pipeline
