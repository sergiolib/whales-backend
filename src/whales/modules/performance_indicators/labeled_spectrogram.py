import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from whales.modules.performance_indicators.performance_indicator import PerformanceIndicator


class LabeledSpectrogram(PerformanceIndicator):
    def __init__(self, logger=None):
        super().__init__(logger)

        self.description = "Labeled Spectrogram"

        self.parameters = {
            "opacity": 0.3,
            "color": "red"
        }

        self.parameters_options = {
            "opacity": [i for i in np.arange(0.1, 1.1, 0.1)],
            "color": ["blue", "red", "greed", "yellow", "magenta", "cyan", "black", "white"]
        }

        self.private_parameters = {
            "data_file": None,
        }

    def method_compute(self):
        df = self.private_parameters["data_file"]
        labels = df.metadata["labels"]
        first = df.data.index[0]
        labels = [((l[0] - first).seconds, (l[1] - l[0]).seconds) for l in labels]
        data = df.data.values.ravel().astype(float)
        D = librosa.amplitude_to_db(librosa.stft(data), ref=np.max)
        plt.figure(figsize=(14, 5))
        axes = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=df.sampling_rate)
        plt.colorbar(format='%+2.0f dB', ticks=np.linspace(-80, 0, 5))
        axes.set_title('Spectrogram')
        axes.set_xlabel('Time [min]')
        axes.set_ylabel('Frequency [Hz]')
        opacity = self.parameters["opacity"]
        color = self.parameters["color"]
        [axes.add_patch(Rectangle((l[0], 0), l[1], 1000, zorder=10, facecolor=color, alpha=opacity)) for l in labels]
        plt.tight_layout()
        return plt.gcf()


PipelineMethod = LabeledSpectrogram
