"""Computing heavy instructions for generating step results throughout the pipeline"""

import logging

from whales.modules.pipelines.getters import get_available_datasets, get_available_datafiles, get_available_formatters, \
    get_available_labels_formatters


class InstructionSet:
    def __init__(self, logger=None):
        self.logger = logger
        if self.logger is None:
            logging.getLogger(self.__class__.__name__)


class SupervisedWhalesInstructionSet(InstructionSet):
    def build_data_set(self, params:  dict):
        available_data_sets = get_available_datasets()
        available_data_files = get_available_datafiles()
        available_formatters = get_available_formatters()

        ds = available_data_sets[params["data_set_type"]["method"]]()

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

    def set_labels(self, params:  dict):
        labels_params = params["input_labels"]
        ds = params["data_set"]

        lf = get_available_labels_formatters()

        for p in labels_params:
            file_name = p["labels_file"]
            labels_formatter = lf[p["labels_formatter"]]()
            for df in ds.datafiles:
                df.load_labels(file_name, labels_formatter, label="whale")

        return {}

    def add_features_extractor(self, params:  dict):
        added_features_extractors = params.get("features_extractors", [])
        added_features_extractors.append(params["features_extractor"])
        return {"features_extractors": added_features_extractors}

    def add_performance_indicator(self, params:  dict):
        added_performance_indicators = params.get("performance_indicators", [])
        added_performance_indicators.append(params["performance_indicator"])
        return {"performance_indicators": added_performance_indicators}

    def add_pre_processing_method(self, params:  dict):
        added_pp_method = params.get("pre_processing_methods", [])
        added_pp_method.append(params["pp_method"])
        return {"pre_processing_methods": added_pp_method}

    def set_machine_learning_method(self, params:  dict):
        return {"ml_method": params["ml_method"]}

    def set_data_iterators(self, params: dict):
        ds = params["data_set"]
        return {
            "training_sets": ds.get_training(),
            "testing_sets": ds.get_testing(),
            "validation_sets": ds.get_validation(),
            "number_of_sets": ds.iterations
        }

    def train_machine_learning_method(self, params: dict):
        return {}

    def train_performance_indicators(self, params: dict):
        return {}

    def train_pre_processing(self, params: dict):
        return {}

    def train_features(self, params: dict):
        return {}

    def transform_features(self, params: dict):
        return {}

    def transform_pre_processing(self, params: dict):
        return {}

    def predict_machine_learning_method(self, params: dict):
        return {}

    def compute_performance_indicators(self, params: dict):
        return {}

    def train_execute_methods(self, params: dict):
        training_sets = params["training_sets"]
        testing_sets = params["testing_sets"]
        validation_sets = params["validation_sets"]

        number_of_sets = params["number_of_sets"]

        # All iterations results dictionary
        results = {}

        # Iterate on sets
        for iteration, (current_training, current_testing, current_validation) in enumerate(zip(
                training_sets, testing_sets, validation_sets)):
            # Set current sets
            current_data_set = {}
            current_data_set["current_training_set"] = current_training
            current_data_set["current_testing_set"] = current_testing
            current_data_set["current_validation_set"] = current_validation

            # Partial results dictionary
            output = {}

            # Train pre processing methods
            output.update(self.train_pre_processing(current_data_set))

            # Train features extractors
            output.update(self.train_features(output))

            # Transform data
            output.update(self.transform_pre_processing(output))
            output.update(self.transform_features(output))

            # Train machine learning method
            output.update(self.train_machine_learning_method(output))

            # Train performance indicators
            output.update(self.train_performance_indicators(output))

            # Predict with machine learning method
            output.update(self.predict_machine_learning_method(output))

            # Compute performance indicators
            output.update(self.compute_performance_indicators(output))

            # Store results
            results[f"{iteration + 1}/{number_of_sets}"] = output

        return results
