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
            "labels": "predictions"
        }

        self.parameters_options = {
            "opacity": ["{:.2}".format((i+1)/10.0) for i in range(10)],
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
        colors = ["b", "r", "g", "y", "c", "m", "k"]
        if self.parameters["labels"] == "predictions":
            labels = self.private_parameters["prediction"]
            for i, l in enumerate(np.unique(labels)):
                c = colors[i]
                curr = D[labels == l]
                if len(curr) > 0:
                    axes.scatter(curr[:, 0], curr[:, 1], c=c, label=f"Label {l}")
            axes.set_title("T-SNE map of predicted labels")
        elif self.parameters["labels"] == "true_labels":
            labels = self.private_parameters["target"].astype(int)
            for i, l in enumerate(np.unique(labels)):
                c = colors[i]
                curr = D[labels == l]
                if len(curr) > 0:
                    axes.scatter(curr[:, 0], curr[:, 1], c=c, label=f"Label {l}")
            axes.set_title("T-SNE map of original labels")
        else:
            axes.scatter(D[:, 0], D[:, 1])
            axes.set_title("T-SNE map of unlabeled data frames")
        axes.set_xlabel('D1')
        axes.set_ylabel('D2')
        plt.legend()
        plt.tight_layout()
        return fig


PipelineMethod = TSNE
