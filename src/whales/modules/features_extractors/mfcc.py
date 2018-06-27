from whales.modules.features_extractors.feature_extraction import FeatureExtraction
from whales.modules.features_extractors.spectral_frames import SpectralFrames
import librosa


class MFCC(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.needs_fitting = False
        self.description = """Mel Frequency Cepstral Coefficients"""
        self.parameters = {
            "win": 2048,
            "step": 1024,
            "n_mfcc": 30,
            "rate": 2000
        }

    def method_transform(self, data):
        """
        :param data: {numpy array} audio recording
        :return: {numpy array} Mel frequency cepstral coefficients
        """

        # Commented until output has correct shape (1 column per sample, multiple rows)

        sfr = SpectralFrames()
        spectral_frames = sfr.transform(data=data)
        melspect = librosa.feature.melspectrogram(S=spectral_frames.data.values.T)
        mfcc = librosa.feature.mfcc(
            S=melspect,
            sr=self.parameters["rate"],
            n_mfcc=self.parameters["n_mfcc"]
        )
        # print(mfcc.shape)
        return mfcc


# PipelineMethod = MFCC
