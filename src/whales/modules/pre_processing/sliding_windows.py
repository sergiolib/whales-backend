from whales.modules.pre_processing.pre_processing import PreProcessing


class SlidingWindows(PreProcessing):
    def __init__(self, logger=None):  # There should be no parameters here
        super().__init__(logger)

        self.needs_fitting = False

        self.description = "Sliding windows from time frames"

        self.parameters = {
            "window_width_seconds": 1.0,  # In seconds
            "overlap": 0.3,
        }

        self.parameters_options = {
            "overlap": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        }

    def method_transform(self):
        data_file = self.all_parameters["data_file"]
        data_file.metadata["window_width"] = self.parameters["window_width_seconds"]
        data_file.metadata["overlap"] = self.parameters["overlap"]
        return data_file


PipelineMethod = SlidingWindows
