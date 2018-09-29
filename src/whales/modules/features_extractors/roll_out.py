from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction
import numpy as np
import librosa
import pandas as pd


class SpectralRollOut(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.needs_fitting = False
        self.description = """Spectral Roll Off"""
        self.parameters = {
            "c": 0.9
        }

    def method_transform(self):
        data_file = self.all_parameters["data_file"]
        if "window_width" not in data_file.metadata:
            self.logger.error("Window width was not specified")
            raise AttributeError
        if "overlap" not in data_file.metadata:
            self.logger.error("Overlap was not specified")
            raise AttributeError
        signal = data_file.data.values.astype(float)
        fs = data_file.sampling_rate
        win = int(data_file.metadata["window_width"] * fs)
        step = int(win * (1.0 - data_file.metadata["overlap"]))
        d = np.abs(librosa.stft(signal, win, hop_length=step, center=False)).T
        eps = 1e-6

        d2 = d ** 2
        total_energy = np.sum(d2, axis=1)
        fft_len = d.shape[1]
        c = self.parameters["c"]
        thres = c * total_energy
        cumsum = np.cumsum(d2, axis=1) + eps
        a = np.nonzero(cumsum > thres.reshape(-1, 1))
        b = np.sum(cumsum > thres.reshape(-1, 1), axis=1)
        res = np.zeros_like(b, dtype=float)
        first_nonzero = []
        last_i = -1
        for i, j in zip(a[0], a[1]):
            if last_i == i:
                continue
            first_nonzero.append(j)
            last_i = i

        res[b > 0] = np.array(first_nonzero) / fft_len

        indexes = data_file.data.index[np.arange(0, len(res) * step, step)]
        fdf = FeatureDataFile("centroid_spread")
        fdf._data = pd.DataFrame({"flux": res}, index=indexes)
        return fdf


PipelineMethod = SpectralRollOut
