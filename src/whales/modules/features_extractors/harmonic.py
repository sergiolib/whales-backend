from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction
import numpy as np
import librosa
import pandas as pd


class SpectralHarmonic(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.needs_fitting = False
        self.description = """Spectral Harmoic"""
        self.parameters = {}

    def method_transform(self):
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
        d = np.abs(librosa.stft(signal, win, hop_length=step, center=False)).T
        eps = 1e-6

        M = int(np.round(0.016 * fs) - 1)
        R = [np.correlate(frame, frame, mode='full') for frame in d]
        R = np.vstack(R)

        g = R[:, d.shape[1] - 1]
        R = R[:, d.shape[1]:-1]

        # estimate m0 (as the first zero crossing of R)
        a = np.nonzero(np.diff(np.sign(R), axis=1))

        if len(a[0]) == 0:
            m0 = R.shape[1] - 1
        else:
            m0 = a[1]
        if M > R.shape[1]:
            M = R.shape[1] - 1
        M = int(M)
        Gamma = np.zeros((len(d), M), dtype=np.float64)
        CSum = np.cumsum(d ** 2, axis=1)
        Gamma[:, m0:M] = R[:, m0:M] / (np.sqrt((g.reshape(-1, 1) * CSum[:, M:m0:-1])) + eps)

        ZCR = (np.diff(Gamma, axis=1))

        if ZCR > 0.15:
            HR = 0.0
            f0 = 0.0
        else:
            if Gamma.shape[1] == 0:
                HR = 1.0
                blag = 0.0
                Gamma = np.zeros((M), dtype=np.float64)
            else:
                HR = np.max(Gamma)
                blag = np.argmax(Gamma)

            # Get fundamental frequency:
            f0 = fs / (blag + eps)
            if f0 > 5000:
                f0 = 0.0
            if HR < 0.1:
                f0 = 0.0

        return (HR, f0)

        indexes = data_file.data.index[np.arange(0, len(norm_d) * step, step)][1:]
        fdf = FeatureDataFile("flux")
        fdf._data = pd.DataFrame({"flux": res}, index=indexes)
        return fdf


PipelineMethod = SpectralHarmonic
