import numpy as np
from whales.modules.features_extractors.feature_extraction import FeatureExtraction
from whales.modules.features_extractors.spectral_frames import SpectralFrames
import librosa


class MFCC(FeatureExtraction):
    def __init__(self, rate, parameters={}, logger=None):
        super(MFCC, self).__init__(logger)
        self.description = """Mel Frequency Cepstral Coefficients"""
        self.needs_fitting = False
        self.parameters = {
            "win": parameters.get("win", 2048),
            "step": parameters.get("step", 1024),
            "n_mfcc": parameters.get("n_mfcc", 30),
            "rate": rate
        }

    def method_transform(self, data):
        """
        :param data: {numpy array} audio recording
        :return: {numpy array} Mel frequency cepstral coefficients
        """

        spectral_frames = SpectralFrames(parameters=self.parameters).transform(data=data)
        melspect = librosa.feature.melspectrogram(S=spectral_frames.T)
        mfcc = librosa.feature.mfcc(
            S=melspect,
            sr=self.parameters["rate"],
            n_mfcc=self.parameters["n_mfcc"]
        )
        print(mfcc.shape)
        return mfcc

PipelineMethod = MFCC
