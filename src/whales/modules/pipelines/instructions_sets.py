"""Computing heavy instructions for generating step results throughout the pipeline"""

import logging
import pandas as pd
from os.path import join

from whales.modules.data_files.audio import AudioDataFile
from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.pipelines import getters


class InstructionSet:
    def __init__(self, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)


class SupervisedWhalesInstructionSet(InstructionSet):
    def set_params(self, params: dict):
        return params

    def build_data_file(self, params:  dict):
        """"""
        available_data_files = getters.get_available_data_files()
        available_formatters = getters.get_available_formatters()

        # Load every small input data file and concatenate all into the big data file
        dfs = []
        for elem in params["input_data"]:
            self.logger.info(f"Loading and appending file {elem['file_name']}")
            file_name = elem["file_name"]
            data_file_name = elem["data_file"]
            formatter_name = elem["formatter"]
            df = available_data_files[data_file_name]()
            fmt = available_formatters[formatter_name]()
            df.load(file_name=file_name, formatter=fmt)
            dfs.append(df)
        big_df = AudioDataFile().concatenate(dfs)

        return {"input_data": big_df}

    def set_labels(self, params:  dict):
        labels_params = params["input_labels"]
        input_data = params["input_data"]

        lf = getters.get_available_labels_formatters()

        for p in labels_params:
            self.logger.info(f"Setting labels in file {p['labels_file']}")
            file_name = p["labels_file"]
            labels_formatter = lf[p["labels_formatter"]]()
            input_data.load_labels(file_name, labels_formatter, label="whale")

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
        ml_method = params["ml_method"]
        df = params["transformed_training_set"]
        self.logger.info(f"Training method {params['ml_method'].__class__.__name__} with {len(df.data)} data points")
        ml_method.parameters["data"] = df
        ml_method.fit()
        return {}

    def train_performance_indicators(self, params: dict):
        pi = params["performance_indicators"]
        df = params["training_set"]
        for p in pi:
            self.logger.info(f"Training performance indicator {p.__class__.__name__} with {len(df.data)} data points")
            p.fit(df)
        return {}

    def train_features(self, params: dict):
        feat = params["features_extractors"]
        df = params["training_set"]
        for f in feat:
            f.parameters["data"] = df
            self.logger.info(f"Training features extractor {f.__class__.__name__} with {len(df.data)} data points")
            f.fit()
        return {}

    def transform_features(self, params: dict):
        feat = params["features_extractors"]
        current_set = {}
        transformed_set = {}
        available_sets = [i for i in params if i.endswith("_set")]
        ret = {}
        for s in available_sets:
            df = current_set[s] = params[s]
            transformed_set[s] = []
            for f in feat:
                f.parameters["data"] = df
                msg = f"Transforming features extractor {f.__class__.__name__} with {len(df.data)} data points " \
                      f"for {s} set"
                self.logger.info(msg)
                res = f.transform()
                transformed_set[s].append(res)

            transformed_set[s] = FeatureDataFile().concatenate(transformed_set[s])
            transformed_set[s].data.index = current_set[s].data.index
            labels = current_set[s].metadata["labels"]
            transformed_set[s].metadata["labels"] = labels

            ret["transformed_" + s] = transformed_set[s]
        return ret

    def transform_pre_processing(self, params: dict):
        pre_processing_methods = params["pre_processing_methods"]
        input_data = params["input_data"]
        data = input_data
        for pp in pre_processing_methods:
            pp.parameters["data"] = data
            self.logger.info(f"Applying pre processing {pp.__class__.__name__} to {len(data.data)} data points")
            data = pp.transform()
        return {"input_data": data}

    def predict_machine_learning_method(self, params: dict):
        ml_method = params["ml_method"]
        results = {}
        for dset in ["training", "testing", "validation"]:
            df = params[f"transformed_{dset}_set"]
            ml_method.parameters["data"] = df
            msg = f"Predicting method {ml_method.__class__.__name__} to {len(df.data)} data points of {dset} set"
            self.logger.info(msg)
            prediction = ml_method.predict()
            results[f"prediction_{dset}"] = prediction
        return results

    def compute_performance_indicators(self, params: dict):
        pi = params["performance_indicators"]
        ns = params["number_of_sets"]
        results = {}
        available_sets = [i for i in params if i.endswith("_set") and not i.startswith("transformed_")]
        for dset in available_sets:
            predicted_labels = None
            target_labels = None
            for i in range(ns):
                this_run_params = params[f"{i + 1}/{ns}"]
                df = this_run_params[f"transformed_" + dset]
                if "prediction_" + dset in this_run_params:
                    if predicted_labels is None:
                        predicted_labels = pd.Series(this_run_params[f"prediction_" + dset])
                    else:
                        predicted_labels = predicted_labels.append(pd.Series(this_run_params[f"prediction_" + dset]))
                if "labels" in df.metadata:
                    if target_labels is None:
                        target_labels = df.metadata["labels"]
                    else:
                        target_labels = target_labels.append(df.metadata["labels"])
            for i in pi:
                i.parameters = {
                    "target": target_labels,
                    "prediction": predicted_labels,
                }
                self.logger.info(f"Performance indicators {i.__class__.__name__} of results from {dset}")
                res = i.compute()
                results[f"pi_{i.__class__.__name__}_{dset}"] = res
        return results

    def build_data_set(self, params: dict):
        self.logger.info("Building data set")
        available_data_sets = getters.get_available_data_sets()
        method = params["ds_options"]["method"]
        ds_cls = available_data_sets[method]
        ds = ds_cls()
        data_file = params["input_data"]
        ds.add_data_file(data_file)
        data_generator = ds.get_data_sets()
        number_of_sets = ds.iterations
        return {"data_generator": data_generator, "number_of_sets": number_of_sets}

    def train_execute_methods(self, params: dict):
        data_generator = params["data_generator"]

        number_of_sets = params["number_of_sets"]

        # All iterations results dictionary
        results = {}

        # Iterate on sets
        for iteration, data in enumerate(data_generator):
            training, testing, validation = data

            # Set current sets
            params["training_set"] = training
            params["testing_set"] = testing
            params["validation_set"] = validation

            # Train features extractors
            params.update(self.train_features(params))

            # Transform data
            params.update(self.transform_features(params))

            # Train machine learning method
            params.update(self.train_machine_learning_method(params))

            # Train performance indicators
            # params.update(self.train_performance_indicators(params))

            # Predict with machine learning method
            params.update(self.predict_machine_learning_method(params))

            # Store results
            results[f"{iteration + 1}/{number_of_sets}"] = params.copy()

        # Compute performance indicators
        params.update(self.compute_performance_indicators({**results, **params}))

        return results

    def train_methods(self, params: dict):
        data_generator = params["data_generator"]

        results = dict()

        # Iterate on single set
        for iteration, training in enumerate(data_generator):
            # Set current sets
            params["training_set"] = training

            # Train features extractors
            params.update(self.train_features(params))

            # Transform data
            params.update(self.transform_features(params))

            # Train machine learning method
            params.update(self.train_machine_learning_method(params))

            # Train performance indicators
            # params.update(self.train_performance_indicators(params))

            # Store results
            results["1/1"] = params.copy()

        # Compute performance indicators
        self.compute_performance_indicators({**results, **params})

        # Save ml_method
        ml_method = params["ml_method"]
        dir = params["trained_models_directory"]
        ml_method.save(join(dir, "ml_model.mdl"))

        return results
