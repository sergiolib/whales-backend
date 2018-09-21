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
            "n_components": 30,
            "window_size": 2000,
            "overlap": 0.3
        }

    def method_transform(self):
        """
        :param data: {numpy array} audio recording
        :return: {numpy array} Mel frequency cepstral coefficients
        """
        data_file = self.all_parameters["data_file"]
        signal = data_file.data.values.astype(float)
        win = self.parameters["window_size"]
        step = int(win * (1.0 - self.parameters["overlap"]))
        fs = data_file.sampling_rate
        d = librosa.stft(signal, win, hop_length=step, center=False)
        indexes = pd.date_range(data_file.data.index[0], data_file.data.index[-1], periods=len(signal)//step)
        melspec = librosa.feature.melspectrogram(S=np.abs(d) ** 2)
        n_coef = self.parameters["n_components"]
        mfccf = mfcc(S=melspec, sr=fs, n_mfcc=n_coef)
        fdf = FeatureDataFile("mfcc")
        fdf.data = mfccf.T
        fdf._data.index = indexes
        return fdf


PipelineMethod = MFCC
