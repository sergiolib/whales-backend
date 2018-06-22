import logging
from glob import glob

from whales.modules.pipelines.getters import get_available_features_extractors, get_available_performance_indicators, \
    get_available_pre_processing


class Loader:
    def __init__(self, pipeline, instructions_set, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.loaders_execution_order = []
        self.pipeline = pipeline
        self.instructions_set = instructions_set


class SupervisedWhalesDetectorLoaders(Loader):
    def __init__(self, pipeline, instructions_set, logger=None):
        super().__init__(pipeline, instructions_set, logger)

        self.loaders_execution_order = [
            self.load_input_data,
            self.load_labels,
            self.load_pre_processing,
            self.load_features_extractors,
            self.load_performance_indicators
        ]

    def load_input_data(self):
        """Add instructions to load the input data. Also, use glob where stars are detected"""
        real_input_data = []
        for elem in self.pipeline.parameters["input_data"]:
            if "file_name" in elem:
                if "*" in elem["file_name"]:
                    # Glob detected
                    file_names = glob(elem["file_name"])
                    data_files = [elem["data_file"]] * len(file_names)
                    formatters = [elem["formatter"]] * len(file_names)
                    new_elems = [{
                        "file_name": a,
                        "data_file": b,
                        "formatter": c,
                    } for a, b, c in zip(file_names, data_files, formatters)]
                    real_input_data += new_elems
                else:
                    real_input_data.append(elem)
        data_set_type = self.pipeline.parameters.get("data_set_type", "files_fold")
        self.pipeline.add_instruction(self.instructions_set.build_data_set, {
            "input_data": real_input_data,
            "data_set_type": data_set_type,
            "sliding_windows": {
                "overlap": 0.3,
                "sliding_window_width": "60s",
            }
        })

    def load_labels(self):
        """Add instructions to load the labels. Also, use glob where stars are detected"""
        real_labels = []
        for elem in self.pipeline.parameters["input_labels"]:
            if "*" in elem["labels_file"]:
                labels_files = glob(elem["labels_file"])
                labels_formatters = [elem["labels_formatter"]] * len(labels_files)
                new_elems = [{
                    "labels_file": a,
                    "labels_formatter": b,
                } for a, b in zip(labels_files, labels_formatters)]
                real_labels += new_elems
            else:
                real_labels.append({
                    elem["labels_file"],
                    elem["labels_formatter"]
                })
        self.pipeline.add_instruction(self.instructions_set.set_labels, {"input_labels": real_labels})

    def load_features_extractors(self):
        """Add instructions to load the features extractors"""
        available_features = get_available_features_extractors()
        for feat in self.pipeline.parameters["features_extractors"]:
            method = feat["method"]
            parameters = feat.get("parameters", {})
            feat_cls = available_features.get(method, None)
            if feat_cls is None:
                raise ValueError(f"{method} is not a correct feature")
            feat_fun = feat_cls()
            feat_fun.parameters = parameters
            self.pipeline.add_instruction(self.instructions_set.add_features_extractor, {"features_extractor": feat_fun})

    def load_pre_processing(self):
        """Add instructions to load the pre processing methods"""
        available_pre_processing = get_available_pre_processing()
        for pp in self.pipeline.parameters["pre_processing"]:
            method = pp["method"]
            parameters = pp.get("parameters", {})
            feat_cls = available_pre_processing.get(method, None)
            if feat_cls is None:
                raise ValueError(f"{method} is not a correct pre processing method")
            feat_fun = feat_cls()
            feat_fun.parameters = parameters
            self.pipeline.add_instruction(self.instructions_set.add_pre_processing_method, {"pp_method": feat_fun})

    def load_performance_indicators(self):
        """Add instructions to load the performance indicators"""
        available_pi = get_available_performance_indicators()
        for pi in self.pipeline.parameters["performance_indicators"]:
            method = pi["method"]
            parameters = pi.get("parameters", {})
            pi_cls = available_pi.get(method, None)
            if pi_cls is None:
                raise ValueError(f"{method} is not a correct performance indicator")
            pi_fun = pi_cls()
            pi_fun.parameters = parameters
            self.pipeline.add_instruction(self.instructions_set.add_performance_indicator,
                                          {"performance_indicator": pi_fun})
