import librosa
import numpy as np

from .feature_extraction import FeatureExtraction


class SpectralFrames(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = """Overlapped frames in frequency domain"""
        self.needs_fitting = False
        self.parameters = {
            "sampling_rate": 2000.0,  # Hz
            "to_db": True  # {boolean} Set to True if you want return stft in decibel scale
        }

    def method_transform(self):
        """
        Transform the audio signal in short fourier fast form for any overlapped frame
        :param data: {numpy array} with audio signal (waveform)
        :return: {numpy array} Contains the short-time fourier transform in [0] axis and frame index in [1] axis
        """
        data = self.parameters["data"]
        f = self.parameters["sampling_rate"]

        data[np.isnan(data)] = 0

        spectrogram = np.abs(np.fft.rfft(data, axis=1)) ** 2
        self.parameters["axis"] = np.fft.rfftfreq(data.shape[1], 1/f)
        if self.parameters["to_db"]:
            spectrogram = 10 * np.log10(spectrogram)
        return spectrogram


PipelineMethod = SpectralFrames
