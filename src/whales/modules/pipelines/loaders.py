"""Methods for generating an instructions pile. Keep them light for letting machine resources go to the instructions
sets"""

import logging
from glob import glob
from os import makedirs
from os.path import join, basename

from whales.modules.pipelines.getters import get_available_features_extractors, get_available_performance_indicators, \
    get_available_pre_processing, get_available_supervised_methods, get_available_unsupervised_methods, \
    get_available_semi_supervised_methods


class Loader:
    def __init__(self, pipeline, instructions_set, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.loaders_execution_order = [self.load_create_directories]
        self.pipeline = pipeline
        self.instructions_set = instructions_set

    def load_create_directories(self):
        """Initial creation of output directories for saving logs, models and results"""
        results_directory = self.pipeline.all_parameters["results_directory"]
        logs_directory = self.pipeline.all_parameters["logs_directory"]
        models_directory = self.pipeline.all_parameters["models_directory"]
        makedirs(results_directory, exist_ok=True)
        makedirs(models_directory, exist_ok=True)
        makedirs(logs_directory, exist_ok=True)
        self.logger.debug("Data, models and results directories created")
        self.pipeline.add_instruction(self.instructions_set.set_params,
                                      {"results_directory": results_directory,
                                       "models_directory": models_directory,
                                       "logs_directory": logs_directory})

    def __repr__(self):
        ret = []
        for loader in self.loaders_execution_order:
            ret.append(str(loader.__name__))
        return "\n".join(ret)


class SupervisedWhalesDetectorLoaders(Loader):
    def __init__(self, pipeline, instructions_set, logger=None):
        Loader.__init__(self, pipeline, instructions_set, logger)

        self.loaders_execution_order += [  # Order matters!
            self.load_input_data,
            self.load_labels,
            self.load_pre_processing,
            self.load_features_extractors,
            self.load_performance_indicators,
            self.load_method,
            self.load_build_data_set,
            self.load_train_execute_methods,
        ]

    def load_input_data(self):
        """Add instructions to load the input data. Also, use glob where stars are detected"""
        real_input_data = []
        for elem in self.pipeline.all_parameters["input_data"]:
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
                self.logger.debug(f"Added file {basename(real_input_data[-1]['file_name'])}")
        self.pipeline.add_instruction(self.instructions_set.build_data_file, {
            "input_data": real_input_data,
        })

    def load_labels(self):
        """Add instructions to load the labels. Also, use glob where stars are detected"""
        real_labels = []
        for elem in self.pipeline.all_parameters["input_labels"]:
            if "*" in elem["labels_file"]:
                labels_files = glob(elem["labels_file"])
                labels_formatters = [elem["labels_formatter"]] * len(labels_files)
                new_elems = [{
                    "labels_file": a,
                    "labels_formatter": b,
                } for a, b in zip(labels_files, labels_formatters)]
                real_labels += new_elems
            else:
                real_labels.append(elem)
        self.pipeline.add_instruction(self.instructions_set.set_labels, {"input_labels": real_labels})

    def load_features_extractors(self):
        """Add instructions to load the features extractors"""
        available_features = get_available_features_extractors()
        for feat in self.pipeline.all_parameters["features_extractors"]:
            method = feat["method"]
            parameters = feat.get("parameters", {})
            feat_cls = available_features.get(method, None)
            if feat_cls is None:
                raise ValueError(f"{method} is not a correct feature")
            feat_fun = feat_cls(logger=self.logger)  # feat_fun.transform() -> datos
            feat_fun.parameters = parameters
            self.pipeline.add_instruction(self.instructions_set.add_features_extractor,
                                          {"features_extractor": feat_fun})

    def load_pre_processing(self):
        """Add instructions to load the pre processing methods"""
        available_pre_processing = get_available_pre_processing()
        for pp in self.pipeline.all_parameters["pre_processing"]:
            method = pp["method"]
            parameters = pp.get("parameters", {})
            feat_cls = available_pre_processing.get(method, None)
            if feat_cls is None:
                raise ValueError(f"{method} is not a correct pre processing method")
            feat_fun = feat_cls(logger=self.logger)
            feat_fun.parameters = parameters
            self.pipeline.add_instruction(self.instructions_set.add_pre_processing_method, {"pp_method": feat_fun})
        self.pipeline.add_instruction(self.instructions_set.transform_pre_processing, {})

    def load_performance_indicators(self):
        """Add instructions to load the performance indicators"""
        available_pi = get_available_performance_indicators()
        for pi in self.pipeline.all_parameters["performance_indicators"]:
            method = pi["method"]
            parameters = pi.get("parameters", {})
            pi_cls = available_pi.get(method, None)
            if pi_cls is None:
                raise ValueError(f"{method} is not a correct performance indicator")
            pi_fun = pi_cls(logger=self.logger)
            pi_fun.parameters = parameters
            self.pipeline.add_instruction(self.instructions_set.add_performance_indicator,
                                          {"performance_indicator": pi_fun})

    def load_method(self):
        """Add instructions to load the machine learning method"""
        method_name = self.pipeline.all_parameters["machine_learning"]["method"]
        method_type = self.pipeline.all_parameters["machine_learning"]["type"]
        method_params = self.pipeline.all_parameters["machine_learning"].get("parameters", {})
        if method_type == "supervised":
            available_methods = get_available_supervised_methods()
        elif method_type == "unsupervised":
            available_methods = get_available_unsupervised_methods()
        elif method_type == "semi_supervised":
            available_methods = get_available_semi_supervised_methods()
        else:
            raise ValueError(f"Machine learning type {method_type} not understood")
        ml_cls = available_methods.get(method_name, None)
        if ml_cls is None:
            raise ValueError(f"Machine learning method {method_name} not understood")
        ml_fun = ml_cls(logger=self.logger)
        ml_fun.parameters = method_params
        self.pipeline.add_instruction(self.instructions_set.set_machine_learning_method,
                                      {"ml_method": ml_fun})

    def load_train_execute_methods(self):
        self.pipeline.add_instruction(self.instructions_set.train_execute_methods, {})

    def load_train_methods(self):
        self.pipeline.add_instruction(self.instructions_set.train_methods, {})

    def load_predict_methods(self):
        self.pipeline.add_instruction(self.instructions_set.predict_methods, {})


class TrainSupervisedWhalesDetectorLoaders(SupervisedWhalesDetectorLoaders):
    def __init__(self, pipeline, instructions_set, logger=None):
        Loader.__init__(self, pipeline, instructions_set, logger)

        self.loaders_execution_order += [
            self.load_input_data,
            self.load_labels,
            self.load_pre_processing,
            self.load_features_extractors,
            self.load_performance_indicators,
            self.load_method,
            self.load_train_methods,
        ]


class PredictSupervisedWhalesDetectorLoaders(SupervisedWhalesDetectorLoaders):
    def __init__(self, pipeline, instructions_set, logger=None):
        Loader.__init__(self, pipeline, instructions_set, logger)

        self.loaders_execution_order += [
            self.load_input_data,
            self.load_labels,
            self.load_pre_processing,
            self.load_features_extractors,
            self.load_performance_indicators,
            self.load_method,
            self.load_predict_methods,
            self.check_trained_models_exist,
        ]

    def load_build_data_set(self):
        data_set_options = {"method": "no_split", "parameters": {}}
        self.pipeline.add_instruction(self.instructions_set.build_data_set, {"ds_options": data_set_options})

    def check_trained_models_exist(self):
        models_directory = self.pipeline.all_parameters["models_directory"]
        trained_models = glob(join(models_directory, "*.mdl"))
        trained_models = [basename(i) for i in trained_models]

        # Check machine learning model
        if "ml_model.mdl" not in trained_models:
            self.logger.error(f"Could not find a machine learning trained model in {models_directory}. "
                              f"You specify a trained model.")
            raise ValueError("Prediction without a trained model")
