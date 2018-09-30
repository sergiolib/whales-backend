from sklearn.manifold import TSNE as SKTSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from whales.modules.performance_indicators.performance_indicator import PerformanceIndicator
import numpy as np


class TSNE(PerformanceIndicator):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "T-SNE"

        self.parameters = {
            "labels": "predictions",
            "colormap": "viridis"
        }

        self.parameters_options = {
            "labels": ["predictions", "true_labels", "None"],
            "colormap": ['viridis', 'plasma', 'inferno', 'magma']
        }

        self.private_parameters = {
            "data_file": None,
        }

    def colors(self, x):
        cmap = plt.get_cmap(self.parameters["colormap"])
        return cmap(x)

    def method_compute(self):
        df = self.private_parameters["features_file"]
        data = df.data.values.astype(float)
        tsne = SKTSNE()
        D = tsne.fit_transform(data)
        fig, axes = plt.subplots(1, 1)
        if self.parameters["labels"] == "predictions":
            labels = self.private_parameters["prediction"]
            uniq = np.unique(labels)
            luniq = len(uniq)
            for i, l in enumerate(uniq):
                c = self.colors(i / luniq)
                curr = D[labels == l]
                if len(curr) > 0:
                    axes.scatter(curr[:, 0], curr[:, 1], c=np.array(c).reshape(1, -1), label=f"Label {l}")
            axes.set_title("T-SNE map of predicted labels")
            plt.legend()
        elif self.parameters["labels"] == "true_labels":
            labels = self.private_parameters["target"].astype(int)
            for i, l in enumerate(np.unique(labels)):
                c = self.colors(i)
                curr = D[labels == l]
                if len(curr) > 0:
                    axes.scatter(curr[:, 0], curr[:, 1], c=np.array(c).reshape(1, -1), label=f"Label {l}")
            axes.set_title("T-SNE map of original labels")
            plt.legend()
        else:
            axes.scatter(D[:, 0], D[:, 1])
            axes.set_title("T-SNE map of unlabeled data frames")
        axes.set_xlabel('D1')
        axes.set_ylabel('D2')
        plt.tight_layout()
        return fig


PipelineMethod = TSNE
