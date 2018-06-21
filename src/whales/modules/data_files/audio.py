import pandas as pd
from whales.modules.data_files.time_series import TimeSeriesDataFile


class AudioDataFile(TimeSeriesDataFile):
    description = "Audio data files"

    def __init__(self, logger=None):
        super(AudioDataFile, self).__init__(logger)
        self.parameters = {
            "sliding_window_width": "60s",  # str
            "overlap": 0.3,  # Percentage
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
            self.logger.error("No duration in unloaded data")
            raise RuntimeError("No duration in unloaded data")
        return duration

    @property
    def data(self):
        data = super(AudioDataFile, self).data
        if "labels" in data.columns:
            return data
        else:
            data["labels"] = self.name_label["unlabeled"]  # Set everything as unlabeled if hasn't been labeled
        return data

    @data.setter
    def data(self, data):
        self._data = data

    def load_labels(self, file_name, labels_formatter, label="whale"):
        label_name = self.parameters["label_mame"]
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

    def generate_sliding_windows(self):
        """Generate one file with fixed number of rows and as many columns needed to store all sliding windows"""
        t0 = self.data.index[0]
        offset = self.parameters["sliding_window_width"]
        offset = pd.Timedelta(offset)
        overlap = offset * self.parameters["overlap"]
        starting_times = [t0]
        finishing_times = []
        while True:  # Just to generate iteration
            # finishing_times.append(starting_times[-1] + offset - 1)  # -1 to avoid errors in case of 0 overlap
            finishing_times.append(starting_times[-1] + offset)
            if finishing_times[-1] > self.data.index[-1]:
                break
            starting_times.append(finishing_times[-1] - overlap)
        for a, b in zip(starting_times, finishing_times):
            data = self.data.loc[a:b]
            new_df = self.__class__()
            new_df.data = data
            new_df.metadata = self.metadata.copy()
            new_df.metadata["num_frames"] = len(data)
            yield new_df


PipelineDatafile = AudioDataFile
