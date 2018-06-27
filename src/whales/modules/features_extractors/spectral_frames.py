import numpy as np
from .feature_extraction import FeatureExtraction
import librosa


class SpectralFrames(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = """Overlapped frames in frequency domain"""
        self.needs_fitting = False
        self.parameters = {
            "win": 2048,  # {int} Sliding windows size. By default is 2024 samples
            "step": 1024,  # {int] Number of points in the sliding windows. By default is 1024 samples
            "to_db": True  # {boolean} Set to True if you want return stft in decibel scale
        }

    def method_transform(self, data):
        """
        Transform the audio signal in short fourier fast form for any overlapped frame
        :param data: {numpy array} with audio signal (waveform)
        :return: {numpy array} Contains the short-time fourier transform in [0] axis and frame index in [1] axis
        """

        # Commented until output has correct shape (1 column per sample, multiple rows)

        # print(self.parameters)
        for column in range(data.shape[1]):
            stft = librosa.stft(data[:, column], self.parameters["win"],
                                hop_length=self.parameters["step"], center=False)
            spectrogram = np.abs(stft) ** 2
            if self.parameters["to_db"]:
                spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max).T
            return spectrogram


# PipelineMethod = SpectralFrames
