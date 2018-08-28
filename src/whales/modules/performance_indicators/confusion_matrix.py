import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from whales.modules.performance_indicators.performance_indicator import PerformanceIndicator


class ConfusionMatrix(PerformanceIndicator):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "Confusion matrix"

        self.parameters = {
            "normalized": False,
            "classes": None,
            "plot": False,
            "title": "Confusion matrix",
            "cmap": "Blues"
        }
        self.private_parameters = {
            "target": None,
            "prediction": None,
        }

    def method_compute(self):
        target = np.array(self.all_parameters["target"])
        prediction = np.array(self.all_parameters["prediction"])
        classes = self.parameters["classes"]
        if classes is None:
            classes = np.unique(target.tolist() + prediction.tolist())
        res = confusion_matrix(target, prediction)
        if self.parameters["normalized"]:
            res = res / len(target)
        if self.parameters["plot"]:
            res = plot_confusion_matrix(cm=res, classes=classes,
                                        title=self.parameters["title"],
                                        cmap=plt.get_cmap(self.parameters["cmap"]))
        return res


def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """

    fig = plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if "float" in str(cm.dtype) else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


PipelineMethod = ConfusionMatrix