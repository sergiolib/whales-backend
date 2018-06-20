import pandas as pd
from whales.modules.datafiles.time_series import TimeSeriesDatafile


class AudioDatafile(TimeSeriesDatafile):
    description = "Audio data files"

    def __init__(self, logger=None):
        super(AudioDatafile, self).__init__(logger)
        self.label_name = {
            # Dictionary that maps actual label index to the label name
            0: "unlabeled"
        }

    @property
    def name_label(self):
        """Inverse of self.label_name"""
        return {b: a for a, b in self.label_name.items()}

    @property
    def duration(self):
        if self.data is not None:
            duration = self.data.index[-1] - self.data.index[0]
        else:
            self.logger.error("No duration in unloaded data")
            raise RuntimeError("No duration in unloaded data")
        return duration

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


PipelineDatafile = AudioDatafile
