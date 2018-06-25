import pandas as pd

from whales.modules.data_files.audio import AudioDataFile
from whales.modules.data_files.audio_windows import AudioWindowsDataFile
from whales.modules.pre_processing.pre_processing import PreProcessing


class SlidingWindows(PreProcessing):
    def __init__(self, logger=None):  # There should be no parameters here
        super().__init__(logger)

        self.needs_fitting = False

        self.description = "Sliding windows from time frames"

        self.parameters = {
            "window_width": "60s",
            "overlap": 0.3,
            "labels_treatment": "mean",
        }

    def method_transform(self, data_file):
        if type(data_file) not in [AudioDataFile]:
            raise ValueError("Input should be a Pandas object")

        t0 = data_file.start_time
        offset = self.parameters["window_width"]
        offset = pd.Timedelta(offset)
        overlap = offset * self.parameters["overlap"]
        starting_times = [t0]
        finishing_times = []

        labels_treatment = self.parameters["labels_treatment"]

        while True:  # Just to generate iteration
            finishing_times.append(starting_times[-1] + offset - 1)
            if finishing_times[-1] > data_file.start_time + data_file.duration:
                break
            starting_times.append(finishing_times[-1] - overlap)
        new_df = AudioWindowsDataFile(data_file)
        for a, b in zip(starting_times, finishing_times):
            if b > data_file.end_time:
                b = data_file.end_time
            labels = data_file.data["labels"].loc[a:b]

            if labels_treatment == "mean":
                label = labels.mean()
            elif labels_treatment == "mode":
                label = labels.mode()
            else:
                raise ValueError("Labels treatment not understood")

            new_df.add_window(a, b, label)
        return new_df


PipelineMethod = SlidingWindows
