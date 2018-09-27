from sklearn.manifold import TSNE as SKTSNE
import matplotlib.pyplot as plt
from whales.modules.performance_indicators.performance_indicator import PerformanceIndicator
import numpy as np


class TSNE(PerformanceIndicator):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "T-SNE"

        self.parameters = {
            "opacity": "0.3",
            "labels_color": "cyan",
            "labels": "predictions"
        }

        self.parameters_options = {
            "opacity": ["{:.2}".format((i+1)/10.0) for i in range(10)],
            "labels_color": ["blue", "red", "green", "yellow", "magenta", "cyan", "black", "white"],
            "labels": ["predictions", "true_labels", "None"]
        }

        self.private_parameters = {
            "data_file": None,
        }

    def method_compute(self):
        df = self.private_parameters["features_file"]
        data = df.data.values.astype(float)
        tsne = SKTSNE()
        D = tsne.fit_transform(data)
        fig, axes = plt.subplots(1, 1)
        if self.parameters["labels"] == "predictions":
            labels = self.private_parameters["prediction"]
            D_true = D[labels]
            D_false = D[~labels]
            axes.scatter(D_true[:, 0], D_true[:, 1], c="b", label="Whale songs")
            axes.scatter(D_false[:, 0], D_false[:, 1], c="r", label="Not whale songs")
            axes.set_title("T-SNE map of labeled data frames")
        elif self.parameters["labels"] == "true_labels":
            labels = self.private_parameters["target"]
            D_true = D[labels]
            D_false = D[~labels]
            axes.scatter(D_true[:, 0], D_true[:, 1], c="b", label="Whale songs")
            axes.scatter(D_false[:, 0], D_false[:, 1], c="r", label="Not whale songs")
            axes.set_title("T-SNE map of labeled data frames")
        else:
            axes.scatter(D[:, 0], D[:, 1])
            axes.set_title("T-SNE map of unlabeled data frames")
        axes.set_xlabel('D1')
        axes.set_ylabel('D2')
        plt.legend()
        plt.tight_layout()
        return fig


PipelineMethod = TSNE
