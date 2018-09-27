from whales.modules.pre_processing.pre_processing import PreProcessing
from whales.modules.data_files.audio import AudioDataFile
import numpy as np


class RejectLowSNR(PreProcessing):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Reject low SNR files"
        self.parameters = {
            "threshold": -6
        }

    def method_transform(self):
        df = self.private_parameters["data_file"]
        if len(df.metadata["labels"]) == 0:
            self.logger.error("Could not apply Reject Low SNR preprocessing: no labels found to compute noise")
            return df
        files = df.metadata["starts_stops"]
        data = df.get_labeled_data()
        ind = data.index
        totalCalls = 0
        totalNoise = 0
        for st, en in files:
            signal = data.loc[st:en]
            labels = signal.pop("labels")
            calls = signal[labels > 0]
            noise = signal[labels == 0]
            if len(calls) == 0:
                continue
            totalCalls += len(calls)
            totalNoise += len(signal)
            snr = self.compute_snr(calls, noise)
            self.logger.info(f"Computed SNR for data between {st} and {en}: {snr:.4}")
            if snr <= self.parameters["threshold"]:
                ind = ind.drop(data[st:en].index)
        self.logger.info(f"Calls / Noise ratio: {totalCalls / totalNoise}")
        new_df = AudioDataFile(df)
        data = df.data.loc[ind]
        if len(data) == 0:
            self.logger.error("All data files picked are too noisy")
            raise RuntimeError("All data files picked are too noisy")
        new_df.data = data
        return new_df

    def compute_snr(self, signal, noise):
        powerCall = np.linalg.norm(signal)
        powerNoise = np.linalg.norm(noise)
        snr = 20 * np.log10(powerCall / powerNoise)
        return snr


PipelineMethod = RejectLowSNR
