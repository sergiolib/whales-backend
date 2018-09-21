import numpy as np
import pandas as pd

from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.features_extractors.feature_extraction import FeatureExtraction


class Range(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Range"
        self.needs_fitting = False
        self.parameters = {}

    def method_transform(self):
        data_file = self.all_parameters["data_file"]
        # Disable using this in settings without sliding windows for now
        if "window_width" not in data_file.metadata:
            self.logger.error("Window width was not specified")
            raise AttributeError
        if "overlap" not in data_file.metadata:
            self.logger.error("Overlap was not specified")
            raise AttributeError

        fs = data_file.sampling_rate
        win = int(data_file.metadata.get("window_width", data_file.duration.seconds) * fs)
        step = int(win * (1.0 - data_file.metadata.get("overlap", 0.0)))
        data = data_file.data.astype(float)
        st = 0
        en = st + step
        res = []
        while True:
            if en > len(data):
                en = len(data)
            row = data.iloc[st:en]
            res.append(row.max() - row.min())
            st = en
            if en == len(data):
                break
            en = en + step
        f = FeatureDataFile("min")
        f.data = np.array(res).reshape(-1, 1)
        inds = pd.date_range(data.index[0], data.index[-1], len(res))
        f.data.index = inds
        f.data = f.data.dropna()
        return f


PipelineMethod = Range
