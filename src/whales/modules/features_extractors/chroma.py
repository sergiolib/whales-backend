from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction
import numpy as np
import librosa
from librosa.feature import mfcc, chroma_stft
import pandas as pd


class Chroma(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.needs_fitting = False
        self.description = """Chroma features"""
        self.parameters = {}

    def method_transform(self):
        """
        :return: {numpy array} Chroma features
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
        chroma = chroma_stft(S=np.abs(d)**2, sr=fs)
        indexes = data_file.data.index[np.arange(0, chroma.shape[1] * step, step)]
        fdf = FeatureDataFile("chroma")
        fdf._data = pd.DataFrame(chroma.T, columns=[f"chroma_{i}" for i in range(chroma.shape[0])], index=indexes)
        return fdf


PipelineMethod = Chroma
