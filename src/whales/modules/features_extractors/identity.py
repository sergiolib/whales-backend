import numpy as np
import pandas as pd
from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Identity(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Identity"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        data_file = self.all_parameters["data_file"]
        if "window_width" not in data_file.metadata:
            self.logger.error("Window width was not specified")
            raise AttributeError
        if "overlap" not in data_file.metadata:
            self.logger.error("Overlap was not specified")
            raise AttributeError
        fs = data_file.sampling_rate
        win = int(data_file.metadata["window_width"] * fs)
        step = int(win * (1.0 - data_file.metadata["overlap"]))
        data = data_file.data.astype(float)
        st = 0
        en = st + step
        res = []
        while True:
            if en > len(data):
                break
            res.append(data.iloc[st:en].values)
            st = en
            en = en + step
        fdf = FeatureDataFile("mfcc")
        fdf.data = np.vstack(res)
        indexes = pd.date_range(data_file.data.index[0], data_file.data.index[-1], periods=len(data_file.data)//step)
        fdf._data.index = indexes
        return fdf


PipelineMethod = Identity
