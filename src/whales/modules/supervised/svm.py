from sklearn.svm import SVC

from whales.modules.sklearn_models import SKLearnSaveLoadMixin
from whales.modules.supervised.supervised import Supervised


class SVM(SKLearnSaveLoadMixin, Supervised):
    def __init__(self, logger=None):
        Supervised.__init__(self, logger)

        self.needs_fitting = True

        self.description = "Support Vector Machine Classifier"

        self._model = SVC()

        self.parameters = {
            "C": 1.0,
            "kernel": "rbf",
            "degree": 3,
            "gamma": "auto",
            "coef0": 0.0,
            "shrinking": True,
            "probability": False,
            "tol": 0.001,
            "cache_size": 200,
            "class_weight": None,
            "verbose": False,
            "max_iter": -1,
            "decision_function_shape": "ovr",
            "random_state": None,
        }

        self.parameters_options = {
            "decision_function_shape": ["ovr", "ovo"],
            "kernel": ["linear", "poly", "rbf", "sigmoid"]
        }

    def method_fit(self):
        data = self.parameters["data"]
        target = self.parameters["target"]
        self._model.fit(data, target)

    def method_predict(self):
        data = self.parameters["data"]
        return self._model.predict(data)


PipelineMethod = SVM
