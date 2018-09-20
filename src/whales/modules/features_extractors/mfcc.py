from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction
from whales.modules.features_extractors.spectrogram import SpectralFrames
import librosa
import pandas as pd


class MFCC(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.needs_fitting = False
        self.description = """Mel Frequency Cepstral Coefficients"""
        self.parameters = {
            "n_components": 30,
        }

    def method_transform(self):
        """
        :param data: {numpy array} audio recording
        :return: {numpy array} Mel frequency cepstral coefficients
        """
        data_file = self.all_parameters["data_file"]
        sfr = SpectralFrames()
        sfr.private_parameters["data_file"] = data_file
        spectral_frames = sfr.method_transform()
        melspect = librosa.feature.melspectrogram(S=spectral_frames.data.values.T)
        mfcc = librosa.feature.mfcc(
            S=melspect,
            sr=list(data_file.metadata["sampling_rate"].items())[0][1],
            n_mfcc=self.parameters["n_components"]
        ).T
        fdf = FeatureDataFile()
        fdf._data = pd.DataFrame(mfcc, index=spectral_frames.data.index, columns=[f"mfcc_{i}" for i in range(mfcc.shape[1])])
        return fdf


PipelineMethod = MFCC
