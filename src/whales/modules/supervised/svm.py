from sklearn.preprocessing import scale
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
            "scale": True
        }

        self.parameters_options = {
            "decision_function_shape": ["ovr", "ovo"],
            "kernel": ["linear", "poly", "rbf", "sigmoid"]
        }

        self.private_parameters = {
            "random_state": None,
        }

    def method_fit(self):
        data = self.all_parameters["data"].values
        data = scale(data) if self.parameters["scale"] else data

        target = self.all_parameters["target"].values.ravel()

        try:
            self._model.fit(data, target)
        except ValueError as e:
            self.logger.error(e)
            raise e

    def method_predict(self):
        data = self.all_parameters["data"]
        data = scale(data) if self.parameters["scale"] else data

        return self._model.predict(data)


PipelineMethod = SVM
