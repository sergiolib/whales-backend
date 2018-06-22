from sklearn.svm import SVC

from whales.modules.sklearn_models import SKLearnSaveLoadMixin
from whales.modules.supervised.supervised import Supervised


class SVM(Supervised, SKLearnSaveLoadMixin):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.needs_fitting = True

        self.description = "This is a demo"

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

    def method_fit(self, data, target):
        self._model.fit(data, target)

    def method_predict(self, data):
        self._model.predict(data)


PipelineMethod = SVM
