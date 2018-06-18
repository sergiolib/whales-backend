import pandas as pd
from whales.modules.datafiles.time_series import TimeSeriesDatafile


class AudioDatafile(TimeSeriesDatafile):
    description = "Audio data files"

    def __init__(self, datafile=None, logger=None):
        super(AudioDatafile, self).__init__(datafile, logger)
        self.label_name = {
            0: "unlabeled"
        }

    @property
    def name_label(self):
        return {b: a for a, b in self.label_name.items()}

    @property
    def duration(self):
        # return self.metadata["num_frames"] / self.metadata["frame_rate"]
        return self.data.index[-1] - self.data.index[0]

    @property
    def data(self):
        data = super(AudioDatafile, self).data
        if "labels" in data.columns:
            return data
        else:
            data["labels"] = self.name_label["unlabeled"]  # Set everything as unlabeled if hasn't been labeled
        return data

    @data.setter
    def data(self, data):
        self._data = data

    def load_labels(self, file_name, labels_formatter, label="whale"):
        if label not in self.name_label:
            self.label_name[min(list(self.label_name.keys())) + 1] = label
        labels = labels_formatter.read(file_name)
        data = self.data
        first = data.index[0]
        for _, row in labels.iterrows():
            start_time = row["Begin_time"]
            end_time = row["End_time"]
            from_first = pd.Timedelta(seconds=start_time)
            delta = pd.Timedelta(seconds=end_time-start_time)
            a = first + from_first
            b = first + from_first + delta
            data[a:b].labels = self.name_label[label]
        self.data = data



PipelineDatafile = AudioDatafile
