import numpy as np
from whales.modules.performance_indicators.performance_indicator import PerformanceIndicator


class Accuracy(PerformanceIndicator):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "Number of correct predictions divided in the total of predicted values"

        self.private_parameters = {
            "target": None,
            "prediction": None,
        }
        self.parameters = {}

    def method_compute(self):
        target = np.array(self.all_parameters["target"])
        prediction = np.array(self.all_parameters["prediction"])
        return np.count_nonzero(target == prediction) / len(target)


PipelineMethod = Accuracy
