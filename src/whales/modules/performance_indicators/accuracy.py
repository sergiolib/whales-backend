import numpy as np
from whales.modules.performance_indicators.performance_indicator import PerformanceIndicator


class Accuracy(PerformanceIndicator):
    def __init__(self, logger=None):
        super(Accuracy, self).__init__(logger)

        self.description = "Number of correct predictions divided in the total of predicted values"

        self.parameters = {
            "target": None,
            "prediction": None,
        }

    def method_evaluate(self):
        target = self.parameters["target"]
        prediction = self.parameters["prediction"]
        return np.count_nonzero(target == prediction) / len(target)


PipelineMethod = Accuracy
