import logging


class InstructionSet:
    def __init__(self, logger=None):
        self.logger = logger
        if self.logger is None:
            logging.getLogger(self.__class__.__name__)


class SupervisedWhalesInstructionSet(InstructionSet):
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