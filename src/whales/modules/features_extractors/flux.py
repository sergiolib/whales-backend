from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction
import numpy as np
import librosa
import pandas as pd


class SpectralFlux(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.needs_fitting = False
        self.description = """Spectral Flux"""
        self.parameters = {}

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

        sum_d = np.sum(d + eps, axis=1)
        norm_d = d / sum_d.reshape(-1, 1)
        res = np.sum((norm_d[1:] - norm_d[:-1]) ** 2, axis=1)

        indexes = data_file.data.index[np.arange(0, len(norm_d) * step, step)][1:]
        fdf = FeatureDataFile("centroid_spread")
        fdf._data = pd.DataFrame({"flux": res}, index=indexes)
        return fdf


PipelineMethod = SpectralFlux
