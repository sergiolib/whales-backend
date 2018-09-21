import datetime

import pandas as pd

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import dates
from whales.modules.data_sets.no_split import NoSplit
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
        ns = NoSplit()
        ns.add_data_file(df)
        ds = list(ns.get_data_sets())[0]
        start_times = df.private_parameters["start_time"]
        end_times = df.private_parameters["end_time"]
        win_index_to_times = {index: (t0, tf) for index, t0, tf in zip(labels.index, start_times, end_times)}
        trimmed_ds = ds.data.dropna()
        ds = ds.data
        deleted_ind = ds.index.difference(trimmed_ds.index).values
        data = df.data.values.ravel().astype(float)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(data)), ref=np.max)
        plt.figure()
        axes = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=df.sampling_rate, x_coords=pd.date_range(start_times[0], end_times[-1], periods=D.shape[1]))
        plt.colorbar(format='%+2.0f dB', ticks=np.linspace(-80, 0, 5))
        axes.set_title('Labeled spectrogram')
        axes.set_xlabel('Time')
        axes.set_ylabel('Frequency [Hz]')
        opacity = float(self.parameters["opacity"])
        labels_color = self.parameters["labels_color"]
        for lv, l in win_index_to_times.items():
            if lv not in deleted_ind and labels[lv] == 1:
                axes.add_patch(Rectangle((l[0], 0), l[1] - l[0], 1000, zorder=10, facecolor=labels_color, alpha=opacity))
        formatter = dates.DateFormatter('%Y-%m-%d %H:%M:%S')
        axes.xaxis.set_major_formatter(formatter)
        for label in axes.get_xmajorticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")
        plt.tight_layout()
        return plt.gcf()


PipelineMethod = LabeledSpectrogram
