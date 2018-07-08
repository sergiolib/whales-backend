from itertools import cycle
from math import ceil
import numpy as np

from whales.modules.data_files.audio import AudioDataFile
from whales.modules.data_files.feature import AudioSegments
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
            "inds_to_df_and_ind": {},  # Mapper of which data file goes in which window
            "number_of_windows": 0
        }

    @property
    def iterations(self):
        return int(round(ceil(1.0 / self.parameters["testing"])))

    def get_data_sets(self):
        n = self.parameters["number_of_windows"]
        n_tr = int(round(n * self.parameters["training"]))
        n_te = int(round(n * self.parameters["testing"]))
        n_val = int(round(n * self.parameters["validation"]))
        inds = np.random.permutation(n).tolist()
        g_te = cycle(inds)
        g_val = cycle(inds[n_te:] + inds[:n_te])
        g_tr = cycle(inds[n_te + n_val:] + inds[:n_te + n_val])
        for k in range(self.iterations):
            tr, te, val = AudioSegments(), AudioSegments(), AudioSegments()

            # Get the indices for the data sets
            tr_inds = []
            te_inds = []
            val_inds = []

            for i in range(n_tr):
                tr_inds.append(next(g_tr))

            for i in range(n_te):
                te_inds.append(next(g_te))

            for i in range(n_val):
                val_inds.append(next(g_val))

            # Iterate over data files and indices
            for df, it in zip([tr, te, val], [tr_inds, te_inds, val_inds]):
                for i in it:
                    df_ind, window_ind = self.parameters["inds_to_df_and_ind"][i]
                    window, label = self.datafiles[df_ind].get_window(window_ind)
                    window.index -= window.index[0]
                    window.name = f"{df_ind}_{window_ind}"
                    df.add_segment(window, label)
                    df.label_name.update(self.datafiles[df_ind].label_name)
            yield tr, te, val

    def add_data_file(self, data_file):
        if type(data_file) is not AudioDataFile:
            raise AttributeError("Only audio data files are admitted")
        super().add_data_file(data_file)
        for i in range(self.parameters["number_of_windows"],
                       self.parameters["number_of_windows"] + len(data_file.parameters["start_time"])):
            self.parameters["inds_to_df_and_ind"][i] = (len(self.datafiles) - 1, self.parameters["number_of_windows"])
            self.parameters["number_of_windows"] += 1


PipelineDataSet = WindowsFold
