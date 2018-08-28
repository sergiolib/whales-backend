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

    def build_data_file(self, params: dict):
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
            df = available_data_files[data_file_name](logger=self.logger)
            fmt = available_formatters[formatter_name](logger=self.logger)
            df.load(file_name=file_name, formatter=fmt)
            dfs.append(df)
        big_df = AudioDataFile(logger=self.logger).concatenate(dfs)

        return {"input_data": big_df}

    def set_labels(self, params:  dict):
        labels_params = params["input_labels"]
        input_data = params["input_data"]

        lf = getters.get_available_labels_formatters()

        for p in labels_params:
            self.logger.info(f"Setting labels in file {p['labels_file']}")
            file_name = p["labels_file"]
            labels_formatter = lf[p["labels_formatter"]](logger=self.logger)
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

    def train_machine_learning_method(self, params: dict):
        ml_method = params["ml_method"]
        if "transformed_training_set" in params:
            df = params["transformed_training_set"]
        else:
            df = params["input_data"]
        self.logger.info(f"Training method {params['ml_method'].__class__.__name__} with {len(df.data)} data points")
        ml_method.parameters["data"] = df
        ml_method.fit()
        return {}

    def save_trained_ml_method(self, params: dict):
        ml_method = params["ml_method"]
        dir = params["models_directory"]
        ml_method.save(join(dir, "ml_model.mdl"))
        return {}

    def train_performance_indicators(self, params: dict):
        pi = params["performance_indicators"]
        df = params["training_set"]
        for p in pi:
            self.logger.info(f"Training performance indicator {p.__class__.__name__} with {len(df.data)} data points")
            p.fit(df)
        return {}

    def train_features(self, params: dict):
        if "features_extractors" in params:
            feat = params["features_extractors"]
            df = params["training_set"]
            for f in feat:
                f.parameters["data"] = df
                self.logger.info(f"Training features extractor {f.__class__.__name__} with {len(df.data)} data points")
                f.fit()
        return {}

    def save_trained_features_extractors(self, params: dict):
        if "features_extractors" in params:
            feat = params["features_extractors"]
            location = params["models_directory"]
            for i, f in enumerate(feat):
                cur_loc = join(location, f'feature_{i}.mdl')
                self.logger.info(f"Saving features extractor {f.__class__.__name__} to {cur_loc}")
                f.save(cur_loc)
        return {}

    def load_trained_features_extractors(self, params: dict):
        feat = params["features_extractors"]
        location = params["models_directory"]
        for i, f in enumerate(feat):
            cur_loc = join(location, f'feature_{i}.mdl')
            self.logger.info(f"Loading features extractor {f.__class__.__name__} from {cur_loc}")
            f.load(cur_loc)
        return {}

    def transform_features(self, params: dict):
        if "features_extractors" in params:
            feat = params["features_extractors"]
            current_set = {}
            transformed_set = {}
            available_sets = [i
                              for i in params if i.endswith("_set") and not "prediction_" in i and not "transformed_" in i]
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

                transformed_set[s] = FeatureDataFile(logger=self.logger).concatenate(transformed_set[s])
                transformed_set[s].data.index = current_set[s].data.index
                labels = current_set[s].metadata["labels"]
                transformed_set[s].metadata["labels"] = labels

                ret["transformed_" + s] = transformed_set[s]
            return ret
        return {}

    def transform_pre_processing(self, params: dict):
        if "pre_processing_methods" in params:
            pre_processing_methods = params["pre_processing_methods"]
            input_data = params["input_data"]
            data = input_data
            for pp in pre_processing_methods:
                pp.parameters["data"] = data
                self.logger.info(f"Applying pre processing {pp.__class__.__name__} to {len(data.data)} data points")
                data = pp.transform()
            return {"input_data": data}
        else:
            return {"input_data": params["input_data"]}

    def predict_machine_learning_method(self, params: dict):
        ml_method = params["ml_method"]
        results = {}
        available_sets = [i for i in params if i.endswith("_set") and i.startswith("transformed_")]
        for dset in available_sets:
            df = params[dset]
            ml_method.parameters["data"] = df
            msg = f"Predicting method {ml_method.__class__.__name__} to {len(df.data)} data points of {dset}"
            self.logger.info(msg)
            prediction = ml_method.predict()
            results["prediction_" + dset] = prediction
        return results

    def load_trained_machine_learning_method(self, params: dict):
        ml_method = params["ml_method"]
        location = params["models_directory"]
        cur_loc = join(location, 'ml_model.mdl')
        self.logger.info(f"Loading machine learning method {ml_method} from {cur_loc}")
        ml_method.load(cur_loc)
        return {}

    def compute_performance_indicators(self, params: dict):
        pi = params["performance_indicators"]
        ns = params["number_of_sets"]
        results = {}
        available_sets = [i for i in params if i.endswith("_set") and i.startswith("transformed_")]
        for dset in available_sets:
            predicted_labels = None
            target_labels = None
            label_names = {}
            for i in range(ns):
                this_run_params = params[f"{i + 1}/{ns}"]
                df = this_run_params[dset]
                if "prediction_" + dset in this_run_params:
                    if predicted_labels is None:
                        predicted_labels = pd.Series(this_run_params["prediction_" + dset])
                    else:
                        predicted_labels = predicted_labels.append(pd.Series(this_run_params[f"prediction_" + dset]))
                if "labels" in df.metadata:
                    if target_labels is None:
                        target_labels = df.metadata["labels"]
                    else:
                        target_labels = target_labels.append(df.metadata["labels"])
                    label_names.update(df.label_name)
            for i in pi:
                i.parameters = {
                    "target": list(map(lambda x: label_names[x], target_labels)),
                    "prediction": list(map(lambda x: label_names[x], predicted_labels)),
                    "classes": [i[1] for i in label_names.items()]
                }
                self.logger.info(f"Performance indicators {i.__class__.__name__} of results from {dset}")
                res = i.compute()
                results[f"{i.__class__.__name__}_{dset}"] = res
        return results

    def save_computed_performance_indicators(self, params: dict):
        pi = params["performance_indicators"]

        # Save methods and results
        location = params["models_directory"]
        for i, p in enumerate(pi):
            cur_loc = join(location, f'{p}')
            self.logger.info(f"Saving performance indicator {p}")
            p.save(cur_loc)

        return {}

    def save_performance_indicators_results(self, params: dict):
        pi = params["performance_indicators"]

        # Save methods and results
        location = params["results_directory"]
        for i, p in enumerate(pi):
            cur_loc = join(location, f'{p}')
            self.logger.info(f"Saving performance indicator {p}")
            p.save_results(cur_loc)

        return {}

    def build_data_set(self, params: dict):
        self.logger.info("Building data set")
        available_data_sets = getters.get_available_data_sets()
        method = params["ds_options"]["method"]
        ds_cls = available_data_sets[method]
        ds = ds_cls(logger=self.logger)
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

            # Save trained features extractors
            params.update(self.save_trained_features_extractors(params))

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

        # Save ml_method
        params.update(self.save_trained_ml_method(params))

        # Compute performance indicators
        params.update(self.compute_performance_indicators({**results, **params}))

        # Save performance indicators to disk
        params.update(self.save_computed_performance_indicators(params))

        # Save performance indicators results to disk
        params.update(self.save_performance_indicators_results(params))
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

            # Save trained features extractors
            params.update(self.save_trained_features_extractors(params))

            # Transform data
            params.update(self.transform_features(params))

            # Train machine learning method
            params.update(self.train_machine_learning_method(params))

            # Store results
            results["1/1"] = params.copy()

        # Save ml_method
        params.update(self.save_trained_ml_method(params))

        return results

    def predict_methods(self, params: dict):
        data_generator = params["data_generator"]

        results = dict()

        # Iterate on single set
        for iteration, predicting_set in enumerate(data_generator):
            # Set current sets
            params["predicting_set"] = predicting_set

            # Load trained features extractors
            params.update(self.load_trained_features_extractors(params))

            # Load trained machine learning method
            params.update(self.load_trained_machine_learning_method(params))

            # Transform data
            params.update(self.transform_features(params))

            # Train machine learning method
            params.update(self.predict_machine_learning_method(params))

            # Train performance indicators
            # params.update(self.train_performance_indicators(params))

            # Store results
            results["1/1"] = params.copy()

        # Compute performance indicators
        self.compute_performance_indicators({**results, **params})

        # Save performance indicators to disk
        params.update(self.save_computed_performance_indicators(params))

        # Save performance indicators results to disk
        params.update(self.save_performance_indicators_results(params))

        return results
