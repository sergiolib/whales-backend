from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale

from whales.modules.sklearn_models import SKLearnSaveLoadMixin
from whales.modules.unsupervised.unsupervised import Unsupervised


class GMM(SKLearnSaveLoadMixin, Unsupervised):
    def __init__(self, logger=None):
        Unsupervised.__init__(self, logger)

        self.needs_fitting = True

        self.description = "Gaussian Mixture Model"

        self._model = GaussianMixture()

        self.parameters = {
            "n_components": 2,
            "covariance_type": "full",
            "scale": True
        }

        self.parameters_options = {
            "covariance_type": ["full", "tied", "diag", "spherical"]
        }

        self.private_parameters = {}

    def method_fit(self):
        data = self.all_parameters["data"].data.values
        data = scale(data) if self.parameters["scale"] else data

        self._model = GaussianMixture(n_components=self.parameters["n_components"], covariance_type=self.parameters["covariance_type"], n_init=10)

        try:
            self._model.fit(data)
        except ValueError as e:
            self.logger.error(e)
            raise e

    def method_predict(self):
        data = self.all_parameters["data"].data.values
        data = scale(data) if self.parameters["scale"] else data

        return self._model.predict(data)


PipelineMethod = GMM
