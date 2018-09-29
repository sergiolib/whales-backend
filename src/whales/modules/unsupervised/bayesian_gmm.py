from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import scale

from whales.modules.sklearn_models import SKLearnSaveLoadMixin
from whales.modules.unsupervised.unsupervised import Unsupervised


class BayesianGMM(SKLearnSaveLoadMixin, Unsupervised):
    def __init__(self, logger=None):
        Unsupervised.__init__(self, logger)

        self.needs_fitting = True

        self.description = "Bayesian Gaussian Mixture Model"

        self._model = BayesianGaussianMixture()

        self.parameters = {
            "n_components": 2,
            "max_iter": 500,
            "n_init": 3,
            "scale": True
        }

        self.parameters_options = {
        }

        self.private_parameters = {}

    def method_fit(self):
        data = self.all_parameters["data"].data.values
        data = scale(data) if self.parameters["scale"] else data
        self._model = BayesianGaussianMixture(n_components=self.parameters["n_components"],
                                              max_iter=self.parameters["max_iter"],
                                              n_init=self.parameters["n_init"])

        try:
            self._model.fit(data)
            if not self._model.converged_:
                self.logger.error("Bayesian Gaussian Mixture model did not converge")
                raise RuntimeError("Bayesian Gaussian Mixture model did not converge")
        except ValueError as e:
            self.logger.error(e)
            raise e

    def method_predict(self):
        data = self.all_parameters["data"].data.values
        data = scale(data)
        predicted = self._model.predict(data)
        return predicted


PipelineMethod = BayesianGMM
