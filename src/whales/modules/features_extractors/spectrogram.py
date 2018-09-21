import numpy as np
import pandas as pd
from whales.modules.data_files.feature import FeatureDataFile
from .feature_extraction import FeatureExtraction


class SpectralFrames(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = """Overlapped frames in frequency domain"""
        self.needs_fitting = False
        self.parameters = {
            "sampling_rate": 2000.0,  # Hz
            "to_db": True,  # {boolean} Set to True if you want return stft in decibel scale
            "window_size": 2000
        }

    def method_transform(self):
        """
        Transform the audio signal in short fourier fast form for any overlapped frame
        :param data_file: {AudioDataFile} with audio signal (waveform)
        :return: {FeatureDatafile} Contains the short-time fourier transform in [0] axis and frame index in [1] axis
        """
        data_file = self.all_parameters["data_file"]
        f = data_file.sampling_rate
        data = data_file.data
        data.rolling()
        spectrogram = np.abs(np.fft.rfft(data, axis=1)) ** 2
        if self.parameters["to_db"]:
            spectrogram = 10 * np.log10(spectrogram)
        fdf = FeatureDataFile()
        fdf._data = pd.DataFrame(spectrogram, index=fdf.data.index, columns=np.fft.rfftfreq(data.shape[1], 1.0 / list(f.items())[0][1]))
        return fdf


PipelineMethod = SpectralFrames
