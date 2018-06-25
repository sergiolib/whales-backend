import pandas as pd
from whales.modules.data_files.time_series import TimeSeriesDataFile


class AudioDataFile(TimeSeriesDataFile):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.description = "Audio data files"
        self.parameters = {
            "label_name": {
                # Dictionary that maps actual label index to the label name
                0: "unlabeled"
            },
        }

    @property
    def name_label(self):
        """Inverse of self.label_name"""
        label_name = self.parameters["label_name"]
        return {b: a for a, b in label_name.items()}

    @property
    def duration(self):
        if self.data is not None:
            duration = self.data.index[-1] - self.data.index[0]
        else:
            raise RuntimeError("No duration in unloaded data")
        return duration

    @property
    def start_time(self):
        if self.data is not None:
            start_time = self.data.index[0]
        else:
            raise RuntimeError("No start time in unloaded data")
        return start_time

    @property
    def end_time(self):
        if self.data is not None:
            end_time = self.data.index[-1]
        else:
            raise RuntimeError("No end time in unloaded data")
        return end_time

    @property
    def data(self):
        data = super().data
        if "labels" in data.columns:
            return data
        else:
            data["labels"] = self.name_label["unlabeled"]  # Set everything as unlabeled if hasn't been labeled
        return data

    @data.setter
    def data(self, data):
        self._data = data

    def load_labels(self, file_name, labels_formatter, label="whale"):
        label_name = self.parameters["label_name"]
        if label not in self.name_label:
            label_name[min(list(label_name.keys())) + 1] = label
        labels = labels_formatter.read(file_name)
        data = self.data
        first = data.index[0]
        cols = list(labels.columns)
        begin_time_col = [i for i in cols if "Begin" in i][0]  # If contains Begin, then assume begin_time
        end_time_col = [i for i in cols if "End" in i][0]  # If contains End, then assume end_time
        for _, row in labels.iterrows():
            start_time = row[begin_time_col]
            end_time = row[end_time_col]
            from_first = pd.Timedelta(seconds=start_time)
            delta = pd.Timedelta(seconds=end_time-start_time)
            a = first + from_first
            b = first + from_first + delta
            data[a:b].labels = self.name_label[label]
        self.data = data


PipelineDatafile = AudioDataFile
