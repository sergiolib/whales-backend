import numpy as np
import pandas as pd
from whales.modules.features_extractors.feature_extraction import FeatureExtraction
from whales.modules.data_files.feature import FeatureDataFile


class Energy(FeatureExtraction):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Entropy of energy"
        self.needs_fitting = False
        self.parameters = {
            "num_of_short_blocks": 10
        }

    def method_transform(self):
        data_file = self.all_parameters["data_file"]
        # Disable using this in settings without sliding windows for now
        if "window_width" not in data_file.metadata:
            self.logger.error("Window width was not specified")
            raise AttributeError
        if "overlap" not in data_file.metadata:
            self.logger.error("Overlap was not specified")
            raise AttributeError

        eps = 1e-6
        fs = data_file.sampling_rate
        win = int(data_file.metadata.get("window_width", data_file.duration.seconds) * fs)
        step = int(win * (1.0 - data_file.metadata.get("overlap", 0.0)))
        data = data_file.data.astype(float).values
        st = 0
        en = st + step
        frames = []
        l = len(data)
        inds = []
        while True:
            if en > l:
                en = l
            frames.append(data[st:en])
            inds.append(data_file.data.index[st])
            st = en
            if en == l:
                break
            en = en + step

        total_frame_energy = np.array([(i**2).sum() for i in frames])

        len_frames = np.array([len(i) for i in frames])

        sub_win_length = np.array([int(np.floor(ll / self.parameters["num_of_short_blocks"])) for ll in len_frames])

        frames = [f[:enf] for f, enf in zip(
            frames,
            sub_win_length * self.parameters["num_of_short_blocks"]
        )]

        sub_windows = [frames[i].reshape(self.parameters["num_of_short_blocks"], sub_win_length[i]) for i in range(len(frames))]

        sub_frame_energy = np.vstack([np.sum(sws**2, axis=1) for sws in sub_windows]) / (total_frame_energy.reshape(-1, 1) + eps)

        entropy = -np.sum(sub_frame_energy * np.log2(sub_frame_energy + eps), axis=1)

        f = FeatureDataFile("entropy_energy")
        f._data = pd.DataFrame({"entropy_energy": entropy}, index=inds)
        return f


PipelineMethod = Energy
