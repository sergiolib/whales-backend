import pandas as pd
from whales.modules.data_files.time_series import TimeSeriesDataFile


class AudioDataFile(TimeSeriesDataFile):
    def __init__(self, data_file=None, logger=None):
        super().__init__(logger)
        self.description = "Audio data files"
        self.parameters = {
            "label_name": {
                # Dictionary that maps actual label index to the label name
                0: "unlabeled"
            },
            "number_of_windows": 0,
            "start_time": [],
            "end_time": [],
            "labels_treatment": "max",
        }

        self.cache_labeled_data = None

        if data_file is not None:
            self._data = data_file.data.copy()
            self.metadata = data_file.metadata.copy()
            self.parameters = data_file.parameters

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

    def get_labeled_data(self, clean=False):
        if not clean and self.cache_labeled_data is not None:
            return self.cache_labeled_data
        data = super().data.to_frame()
        labels_list = [self.name_label["unlabeled"]] * len(data)
        labels_series = pd.Series(labels_list, index=data.index)
        if "labels" not in self.metadata:
            self.metadata["labels"] = []
        for a, b, l in self.metadata["labels"]:
            labels_series[a:b] = l
        data["labels"] = labels_series
        self.cache_labeled_data = data
        return data

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
        if "labels" not in self.metadata:
            self.metadata["labels"] = []
        for _, row in labels.iterrows():
            start_time = row[begin_time_col]
            end_time = row[end_time_col]
            from_first = pd.Timedelta(seconds=start_time)
            delta = pd.Timedelta(seconds=end_time-start_time)
            a = first + from_first
            b = first + from_first + delta
            self.metadata["labels"].append((a, b, self.name_label[label]))

    def get_windows_data_frame(self):
        sw = []
        for i in range(self.parameters["number_of_windows"]):
            window = self.get_window(i)
            window[0].index -= window[0].index[0]
            sw.append(window[0])
        return pd.concat(sw, axis=1, sort=False).T

    def get_window(self, ind, return_label=True):
        data = super().data
        if self.parameters["number_of_windows"] == 0:
            return data
        n = len(self.parameters["start_time"])
        if ind > n:
            return None
        st = self.parameters["start_time"][ind]
        en = self.parameters["end_time"][ind]
        label = 0
        if return_label:
            labeled_data = self.get_labeled_data()
            labels_treatment = self.parameters["labels_treatment"]
            if labels_treatment == "max":
                label = int(labeled_data.loc[st:en].labels.max())
            elif labels_treatment == "mode":
                label = int(labeled_data.loc[st:en].labels.mode())
            elif labels_treatment == "mean":
                label = labeled_data.loc[st:en].labels.mean()
            else:
                raise ValueError(f"labels_treatment parameter value not understood: {labels_treatment}")
        window = data.loc[st:en]
        window.name += f"_{ind + 1}_{n}"
        if return_label:
            return window, label
        else:
            return window

    def add_window(self, start_time, end_time):
        if not self.start_time <= start_time < end_time <= self.end_time:
            raise AttributeError("Times are not in a correct range")

        self.parameters["number_of_windows"] += 1
        self.parameters["start_time"].append(start_time)
        self.parameters["end_time"].append(end_time)

    def __repr__(self):
        if hasattr(self, "data"):
            n_windows = len(self.parameters["start_time"])
            if n_windows > 0:
                return f"{self.__class__.__name__} ({n_windows} windows)"
            else:
                return f"{self.__class__.__name__} ({self.duration})"
        return f"{self.__class__.__name__}"

    def concatenate(self, datafiles_list):
        """Add data_files from datafiles_list to new datafile and return it"""
        new_df = self.__class__()
        for df in datafiles_list:
            if new_df._data is None:
                new_df.data = df.data.copy()
            else:
                new_df.data.append(df.data)
        new_df.data.sort_index(inplace=True)
        return new_df


PipelineDataFile = AudioDataFile
