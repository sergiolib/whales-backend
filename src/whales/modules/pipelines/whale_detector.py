from whales.modules.pipelines.pipeline import Pipeline
from whales.modules.pipelines.getters import get_available_datasets, get_available_formatters, get_available_datafiles


class WhaleDetector(Pipeline):
    def __init__(self, logger=None):
        super(WhaleDetector, self).__init__(logger)

        self.instruction_set = {
            "build_data_set": self.build_data_set,
            "set_labels": self.set_labels,
        }

        self.parameters = {
            "necessary_parameters": {
                "output_directory": str,
                "pipeline_type": str,
                "input_data": list,
                "input_labels": list,
            },
            "optional_parameters": {
                "pre_processing": list,
                "features_extractors": list,
                "performance_indicators": list,
                "machine_learning": dict,
                "data_set_type": dict,
                "active": bool,
                "verbose": bool,
                "seed": int,
            },
        }

    # Initialize function
    def initialize(self):
        self.load_input_data()
        self.load_labels()
        self.load_features_extractors()
        self.load_performance_indicators()

    # Instructions
    def build_data_set(self, params):
        available_data_sets = get_available_datasets()
        available_data_files = get_available_datafiles()
        available_formatters = get_available_formatters()

        ds = available_data_sets[params["data_set_type"]]()

        for elem in params["input_data"]:
            file_name = elem["file_name"]
            data_file_name = elem["data_file"]
            formatter_name = elem["formatter"]
            df = available_data_files[data_file_name]()
            fmt = available_formatters[formatter_name]()
            df.load_data(file_name=file_name, formatter=fmt)

            # Sliding windows
            sw_params = params.get("sliding_windows")
            if sw_params is not None:
                df.parameters.update(sw_params)
                sliding_windows = df.generate_sliding_windows()
                [ds.add_datafile(i) for i in sliding_windows]
            else:
                ds.add_datafile(df)
        return {"data_set": ds}

    def set_labels(self, params):
        results = params.get("results", {})
        data_set = params.get("data_set")
        return {}


PipelineType = WhaleDetector
