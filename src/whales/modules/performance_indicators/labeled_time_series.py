import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import dates
from whales.modules.performance_indicators.performance_indicator import PerformanceIndicator


class LabeledTimeSeries(PerformanceIndicator):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "Labeled Audio Signal"

        self.parameters = {
            "opacity": "0.3",
            "labels_color": "orange",
            "line_color": "blue",
        }

        self.parameters_options = {
            "opacity": ["{:.2}".format((i+1)/10.0) for i in range(10)],
            "labels_color": ["blue", "red", "green", "yellow", "magenta", "cyan", "black", "white", "orange"],
            "line_color": ["blue", "red", "green", "yellow", "magenta", "cyan", "black", "white", "orange"]
        }

        self.private_parameters = {
            "data_file": None,
        }

    def method_compute(self):
        df = self.private_parameters["data_file"]
        labels = self.private_parameters["prediction"]
        plt.figure()
        inds = df.data.index
        lab_inds = inds[np.linspace(0, len(inds), len(labels), dtype=int, endpoint=False)]
        fig, axes = plt.subplots(1, 1)
        plt.plot(df.data, color=self.parameters["line_color"])
        axes.set_title('Labeled audio signal')
        axes.set_ylabel('Amplitude')
        axes.set_xlabel('Time')
        opacity = float(self.parameters["opacity"])
        labels_color = self.parameters["labels_color"]
        labels = pd.Series(labels, index=lab_inds)
        width = lab_inds[1] - lab_inds[0]
        h = (df.data.min(), df.data.max())
        for i, l in enumerate(labels):
            if l is True:
                axes.add_patch(
                    Rectangle((lab_inds[i], h[0]), width, h[1] - h[0], zorder=10, facecolor=labels_color, alpha=opacity))
        formatter = dates.DateFormatter('%Y-%m-%d %H:%M:%S')
        axes.xaxis.set_major_formatter(formatter)
        for label in axes.get_xmajorticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        plt.tight_layout()
        return plt.gcf()

PipelineMethod = LabeledTimeSeries
