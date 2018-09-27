from sklearn.cluster import DBSCAN as SKDBSCAN
from sklearn.neighbors import NearestNeighbors
from whales.modules.sklearn_models import SKLearnSaveLoadMixin
from whales.modules.unsupervised.unsupervised import Unsupervised


class DBSCAN(SKLearnSaveLoadMixin, Unsupervised):
    def __init__(self, logger=None):
        Unsupervised.__init__(self, logger)

        self.needs_fitting = False

        self.description = "Density Based Spatial Clustering od Applications and Noise"

        self._model = SKDBSCAN()

        self.parameters = {
            "min_samples": "auto"
        }

        self.parameters_options = {}

        self.private_parameters = {}

    def method_predict(self):
        data = self.all_parameters["data"].data.values
        min_points = self.parameters["min_samples"]
        if min_points == "auto":
            min_points = max(1, data.shape[1] - 2)
        nbrs = NearestNeighbors(n_neighbors=min_points)
        nbrs.fit(data)
        k_neigh = nbrs.kneighbors()[0][:, min_points - 1]
        k_mean = k_neigh.mean()
        eps = k_mean
        self._model = SKDBSCAN(eps=eps, min_samples=min_points)

        try:
            labels = self._model.fit_predict(data)
        except ValueError as e:
            self.logger.error(e)
            raise e

        return labels


PipelineMethod = DBSCAN
