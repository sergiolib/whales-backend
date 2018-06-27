from math import ceil
import numpy as np
import pandas as pd

from whales.modules.data_files.audio_windows import AudioWindowsDataFile
from whales.modules.data_files.data_files import DataFile
from whales.modules.data_sets.data_sets import DataSet


class WindowsFold(DataSet):
    """Perform K fold where every window is treated as a separate source"""
    def __init__(self, logger=None):
        super().__init__(logger)
        self.current_file = 0

        self.description = """K Folding of three separate sets for every folding"""

        self.parameters = {
            "training": 0.6,
            "testing": 0.2,
            "validation": 0.2,
            "inds_to_df_and_ind": {},
            "number_of_windows": 0
        }

    @property
    def iterations(self):
        return ceil(1.0 / self.parameters["testing"])

    def get_data_frames(self):
        tr = DataFile()
        te = DataFile()
        val = DataFile()
        n = self.parameters["number_of_windows"]
        n_tr = int(round(n * self.parameters["training"]))
        n_val = int(round(n * self.parameters["validation"]))
        offset_step = int(round(n * self.parameters["testing"]))
        inds_list = np.random.permutation(n).tolist()
        for k in range(self.iterations):
            offset = k * offset_step
            f = inds_list[offset:]
            s = inds_list[:offset]
            inds = f + s
            tr_inds = inds[:n_tr]
            te_inds = inds[n_tr:-n_val]
            val_inds = inds[-n_val:]
            training_windows = []
            testing_windows = []
            validation_windows = []
            for df, it, winds in zip([tr, te, val], [tr_inds, te_inds, val_inds],
                                       [training_windows, testing_windows, validation_windows]):
                for i in it:
                    df_ind, window_ind = self.parameters["inds_to_df_and_ind"][i]
                    window = self.datafiles[df_ind].get_window(window_ind)
                    window.name = None
                    winds.append(window)
                df.data = pd.concat(winds, axis=1, sort=False).T
            yield tr, te, val

    def add_data_file(self, data_file):
        if type(data_file) is not AudioWindowsDataFile:
            raise AttributeError("Only window data files are admitted")
        super().add_data_file(data_file)
        for i in range(self.parameters["number_of_windows"],
                       self.parameters["number_of_windows"] + len(data_file.parameters["start_time"])):
            self.parameters["inds_to_df_and_ind"][i] = (len(self.datafiles) - 1, self.parameters["number_of_windows"])
            self.parameters["number_of_windows"] += 1


PipelineDataSet = WindowsFold
