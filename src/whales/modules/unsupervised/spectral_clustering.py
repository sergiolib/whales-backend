from sklearn.cluster import SpectralClustering as SKSpectralClustering
from whales.modules.sklearn_models import SKLearnSaveLoadMixin
from whales.modules.unsupervised.unsupervised import Unsupervised


class SpectralClustering(SKLearnSaveLoadMixin, Unsupervised):
    def __init__(self, logger=None):
        Unsupervised.__init__(self, logger)

        self.needs_fitting = False

        self.description = "Spectral Clustering of Projection to Normalized Laplacian"

        self._model = SKSpectralClustering()

        self.parameters = {
            "n_components": 6,
            "affinity": "nearest_neighbors"
        }

        self.parameters_options = {
            "affinity": ["nearest_neighbors", "rbf"]
        }

        self.private_parameters = {}

    def method_predict(self):
        data = self.all_parameters["data"].data.values
        n_clust = self.all_parameters["n_components"]
        self._model = SKSpectralClustering(n_clust,
                                           eigen_solver='arpack',
                                           n_neighbors=10,
                                           affinity='nearest_neighbors',
                                           n_jobs=-1,
                                           n_init=10)

        try:
            return self._model.fit_predict(data)
        except ValueError as e:
            self.logger.error(e)
            raise e


PipelineMethod = SpectralClustering
