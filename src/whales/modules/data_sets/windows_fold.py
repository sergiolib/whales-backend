from math import ceil
import numpy as np

from whales.modules.data_files.audio_windows import AudioWindowsDataFile
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

    def get_data_files(self):
        tr = AudioWindowsDataFile()
        te = AudioWindowsDataFile()
        val = AudioWindowsDataFile()
        n = self.parameters["number_of_windows"]
        n_tr = int(round(n * self.parameters["training"]))
        n_val = int(round(n * self.parameters["validation"]))
        inds = np.random.permutation(n)
        tr_inds = inds[:n_tr]
        te_inds = inds[n_tr:-n_val]
        val_inds = inds[-n_val:]
        df = tr
        for i in tr_inds:
            df_ind, window_ind = self.parameters["inds_to_df_and_ind"][i]
            window = self.datafiles[df_ind].get_window(window_ind)
            df.add_window()
        return tr, te, val

    def add_data_file(self, data_file):
        if type(data_file) is not AudioWindowsDataFile:
            raise AttributeError("Only window data files are admitted")
        super().add_data_file(data_file)
        for i in range(self.parameters["number_of_windows"],
                       self.parameters["number_of_windows"] + len(data_file.parameters["start_time"])):
            self.parameters["inds_to_df_and_ind"][i] = (len(self.datafiles) - 1, self.parameters["number_of_windows"])
            self.parameters["number_of_windows"] += 1


PipelineDataSet = WindowsFold
