import pandas as pd
from whales.modules.data_files.time_series import TimeSeriesDataFile


class AudioDataFile(TimeSeriesDataFile):
    def __init__(self, data_file=None, logger=None):
        super().__init__(logger)
        self.description = "Audio data files"
        self.parameters = {}
        self.private_parameters = {
            "start_time": [],
            "end_time": [],
            "labels_treatment": "max",
            "number_of_windows": 0,
        }

        self.label_name = {
            # Dictionary that maps actual label index to the label name
            0: "unlabeled"
        }

        self.cache_labeled_data = None
        self.labeled_data_changed = False

        self.metadata["labels"] = []

        if data_file is not None:
            self._data = data_file.data.copy()
            self.metadata = data_file.metadata.copy()
            self.parameters = data_file.parameters
            self.private_parameters = data_file.private_parameters

    @property
    def name_label(self):
        """Inverse of self.label_name"""
        label_name = self.label_name
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

    def get_labeled_data(self):
        if not self.labeled_data_changed:
            return self.cache_labeled_data
        data = super().data.to_frame()
        labels_list = [self.name_label["unlabeled"]] * len(data)
        labels_series = pd.Series(labels_list, index=data.index)
        for a, b, l in self.metadata["labels"]:
            labels_series[a:b] = l
        data["labels"] = labels_series
        self.cache_labeled_data = data
        self.labeled_data_changed = False
        return data

    def load_labels(self, file_name, labels_formatter, label="whale"):
        self.labeled_data_changed = True
        label_name = self.label_name
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
            self.metadata["labels"].append((a, b, self.name_label[label]))

    def __repr__(self):
        if hasattr(self, "data") and self.data is not None:
            n_windows = self.all_parameters["number_of_windows"]
            if n_windows > 0:
                return f"{self.__class__.__name__} ({n_windows} windows)"
            else:
                return f"{self.__class__.__name__} ({self.duration})"
        return f"{self.__class__.__name__}"

    def concatenate(self, datafiles_list):
        """Add data_files from datafiles_list to new datafile and return it"""
        new_df = self.__class__()
        series_list = []
        for df in datafiles_list:
            series_list.append(df.data)
            new_df.label_name.update(df.label_name)
        new_df.data = pd.concat(series_list)
        new_df.data.sort_index(inplace=True)
        new_df.metadata["starts_stops"] = [(i.index[0], i.index[-1]) for i in series_list]
        new_df.labeled_data_changed = True
        return new_df

    def load(self, file_name: str, formatter):
        res = super().load(file_name=file_name, formatter=formatter)
        self.metadata["starts_stops"] = [(self.data.index[0], self.data.index[-1])]
        self.labeled_data_changed = True
        return res


PipelineDataFile = AudioDataFile
