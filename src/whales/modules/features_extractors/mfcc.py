from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction
import numpy as np
import librosa
from librosa.feature import mfcc
import pandas as pd


class MFCC(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.needs_fitting = False
        self.description = """Mel Frequency Cepstral Coefficients"""
        self.parameters = {
            "n_components": 30
        }

    def method_transform(self):
        """
        :param data: {numpy array} audio recording
        :return: {numpy array} Mel frequency cepstral coefficients
        """
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
        d = librosa.stft(signal, win, hop_length=step, center=False)
        melspec = librosa.feature.melspectrogram(S=np.abs(d) ** 2)
        n_coef = self.parameters["n_components"]
        mfccf = mfcc(S=melspec, sr=fs, n_mfcc=n_coef)
        indexes = data_file.data.index[np.arange(0, mfccf.shape[1] * step, step)]
        fdf = FeatureDataFile("mfcc")
        fdf._data = pd.DataFrame(mfccf.T, columns=[f"mfcc_{i}" for i in range(n_coef)], index=indexes)
        return fdf


PipelineMethod = MFCC
