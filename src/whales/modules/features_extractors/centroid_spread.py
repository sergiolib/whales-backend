from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction
import numpy as np
import librosa
from librosa.feature import mfcc
import pandas as pd


class CentroidSpread(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.needs_fitting = False
        self.description = """Spectral Centroid and Spread"""
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
        inind = np.arange(0, d.shape[1]) + 1
        j = fs / 2.0
        inind = inind * j / d.shape[1]
        d = d / d.max()
        num = np.sum(inind.reshape(1, -1) * d, axis=1)
        den = np.sum(d, axis=1) + eps
        c = num / den
        s = np.sqrt(np.sum(((inind.reshape(1, -1) - c.reshape(-1, 1))**2) * d, axis=1) / den)
        c = c / j
        s = s / j

        indexes = data_file.data.index[np.arange(0, len(c) * step, step)]
        fdf = FeatureDataFile("centroid_spread")
        fdf._data = pd.DataFrame({"centroid": c, "spread": s}, index=indexes)
        return fdf


PipelineMethod = CentroidSpread
