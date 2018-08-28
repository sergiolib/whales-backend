from whales.modules.features_extractors.feature_extraction import FeatureExtraction
from whales.modules.features_extractors.spectrogram import SpectralFrames
import librosa


class MFCC(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.needs_fitting = False
        self.description = """Mel Frequency Cepstral Coefficients"""
        self.parameters = {
            "win": 2048,
            "step": 1024,
            "n_components": 30,
            "sampling_rate": 2000.0
        }

    def method_transform(self):
        """
        :param data: {numpy array} audio recording
        :return: {numpy array} Mel frequency cepstral coefficients
        """
        data = self.all_parameters["data"]
        sfr = SpectralFrames()
        sfr.parameters["data"] = data
        spectral_frames = sfr.method_transform()
        melspect = librosa.feature.melspectrogram(S=spectral_frames.T)
        mfcc = librosa.feature.mfcc(
            S=melspect,
            sr=self.parameters["sampling_rate"],
            n_mfcc=self.parameters["n_components"]
        ).T
        return mfcc


PipelineMethod = MFCC
