import pandas as pd
from whales.modules.data_files.audio import AudioDataFile


class AudioWindowsDataFile(AudioDataFile):
    def __init__(self, data_file=None, logger=None):
        super().__init__(logger)
        self.description = "Audio windows data files"
        self.parameters = {
            "sliding_window_width": "60s",  # str
            "overlap": 0.3,  # Percentage
            "label_name": {
                # Dictionary that maps actual label index to the label name
                0: "unlabeled"
            },
            "number_of_windows": 0,
            "start_time": [],
            "end_time": [],
            "label": [],
        }

        if data_file is not None:
            self._data = data_file.data.copy()
            self.metadata = data_file.metadata.copy()
            self.parameters = data_file.parameters

    def get_windows_data_frame(self):
        sw = []
        for i in range(self.parameters["number_of_windows"]):
            window = self.get_window(i)
            window.name = None
            sw.append(window)
        return pd.concat(sw, axis=1, sort=False).T

    def get_window(self, ind):
        data = super().data
        if self.parameters["number_of_windows"] == 0:
            return data
        if ind > len(self.parameters["start_time"]):
            return None
        st = self.parameters["start_time"][ind]
        en = self.parameters["end_time"][ind]
        label = self.parameters["label"][ind]
        window = data.loc[st:en]
        window = window.reset_index()["data_0"]
        window["labels"] = label
        return window

    def add_window(self, start_time, end_time, label=0):
        if not self.start_time <= start_time < end_time <= self.end_time:
            raise AttributeError("Times are not in a correct range")

        self.parameters["number_of_windows"] += 1
        self.parameters["start_time"].append(start_time)
        self.parameters["end_time"].append(end_time)
        self.parameters["label"].append(label)
