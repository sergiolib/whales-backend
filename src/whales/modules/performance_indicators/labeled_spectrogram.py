import pandas as pd

import librosa
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import dates
from whales.modules.performance_indicators.performance_indicator import PerformanceIndicator


class LabeledSpectrogram(PerformanceIndicator):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "Labeled Spectrogram"

        self.parameters = {
            "opacity": "0.3",
            "labels_color": "cyan"
        }

        self.parameters_options = {
            "opacity": ["{:.2}".format((i+1)/10.0) for i in range(10)],
            "labels_color": ["blue", "red", "green", "yellow", "magenta", "cyan", "black", "white"]
        }

        self.private_parameters = {
            "data_file": None,
        }

    def method_compute(self):
        df = self.private_parameters["data_file"]
        labels = self.private_parameters["prediction"]
        data = df.data.values.astype(float)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
        plt.figure()
        inds = df.data.index
        lab_inds = inds[np.linspace(0, len(inds), len(labels), dtype=int, endpoint=False)]
        inds = inds[np.linspace(0, len(inds), D.shape[1], dtype=int, endpoint=False)]
        axes = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=df.sampling_rate, x_coords=inds)
        plt.colorbar(format='%+2.0f dB', ticks=np.linspace(-80, 0, 5))
        axes.set_title('Labeled spectrogram')
        axes.set_xlabel('Time')
        axes.set_ylabel('Frequency [Hz]')
        opacity = float(self.parameters["opacity"])
        labels_color = self.parameters["labels_color"]
        labels = pd.Series(labels, index=lab_inds)
        width = lab_inds[1] - lab_inds[0]
        for i, l in enumerate(labels):
            if l is True:
                axes.add_patch(Rectangle((lab_inds[i], 0), width, 1000, zorder=10, facecolor=labels_color, alpha=opacity))
        formatter = dates.DateFormatter('%Y-%m-%d %H:%M:%S')
        axes.xaxis.set_major_formatter(formatter)
        for label in axes.get_xmajorticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        plt.tight_layout()
        return plt.gcf()


PipelineMethod = LabeledSpectrogram
