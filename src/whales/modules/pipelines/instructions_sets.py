"""Computing heavy instructions for generating step results throughout the pipeline"""

import logging
from os.path import basename

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


class WhalesInstructionSet(InstructionSet):
    def set_params(self, params: dict):
        return params

    def build_data_file(self, params: dict):
        """"""
        available_data_files = getters.get_available_data_files()
        available_formatters = getters.get_available_formatters()

        # Load every small input data file and concatenate all into the big data file
        dfs = []
        for elem in params["input_data"]:
            self.logger.info(f"Loading and appending file {basename(elem['file_name'])}")
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
            self.logger.info(f"Setting labels in file {basename(p['labels_file'])}")
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
        df = params["training_set"]
        self.logger.info(f"Training method {params['ml_method'].__class__.__name__} with {len(df.data)} data points")
        ml_method.private_parameters["data"] = df
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
            df = params["input_data"]
            for f in feat:
                f.private_parameters["data_file"] = df
                self.logger.info(f"Training features extractor {f.__class__.__name__} with {len(df.data)} data points")
                f.fit()
        self.save_trained_features_extractors(params)
        return {}

    def save_trained_features_extractors(self, params: dict):
        if "features_extractors" in params:
            feat = params["features_extractors"]
            location = params["models_directory"]
            for i, f in enumerate(feat):
                cur_loc = join(location, f'feature_{i}.mdl')
                self.logger.info(f"Saving features extractor {f.__class__.__name__}")
                f.save(cur_loc)
        return {}

    def load_trained_features_extractors(self, params: dict):
        feat = params["features_extractors"]
        location = params["models_directory"]
        for i, f in enumerate(feat):
            cur_loc = join(location, f'feature_{i}.mdl')
            self.logger.info(f"Loading features extractor {f.__class__.__name__}")
            f.load(cur_loc)
        return {}

    def transform_features(self, params: dict):
        if "features_extractors" in params:
            feat = params["features_extractors"]
            transformed = []
            df = params["input_data"]
            for f in feat:
                f.private_parameters["data_file"] = df
                msg = f"Transforming features extractor {f.__class__.__name__} with {len(df.data)} data points "
                self.logger.info(msg)
                res = f.transform()
                transformed.append(res)

            transformed = FeatureDataFile("all_features", logger=self.logger).concatenate(transformed)
            transformed.metadata["labels"] = df.metadata["labels"]
            return {params["data_set_name"]: transformed}
        return {}

    def transform_pre_processing(self, params: dict):
        if "pre_processing_methods" in params:
            pre_processing_methods = params["pre_processing_methods"]
            input_data = params["input_data"]
            data_file = input_data
            for pp in pre_processing_methods:
                pp.private_parameters["data_file"] = data_file
                self.logger.info(f"Applying pre processing {pp.__class__.__name__} to {len(data_file.data)} data points")
                data_file = pp.transform()
            return {"input_data": data_file}
        else:
            return {"input_data": params["input_data"]}

    def predict_machine_learning_method(self, params: dict):
        ml_method = params["ml_method"]
        results = {}
        available_sets = [i for i in params if i.endswith("_set")]
        for dset in available_sets:
            df = params[dset]
            ml_method.private_parameters["data"] = df
            msg = f"Predicting method {ml_method.__class__.__name__} to {len(df.data)} data points of {dset}"
            self.logger.info(msg)
            prediction = ml_method.predict()
            results["prediction_" + dset] = prediction
        return results

    def load_trained_machine_learning_method(self, params: dict):
        ml_method = params["ml_method"]
        location = params["models_directory"]
        cur_loc = join(location, 'ml_model.mdl')
        self.logger.info(f"Loading machine learning method {ml_method}")
        ml_method.load(cur_loc)
        return {}

    def compute_performance_indicators(self, params: dict):
        pi = params["performance_indicators"]
        results = {}
        available_sets = [i for i in params if i.endswith("_set") and not i.startswith("prediction_")]
        for dset in available_sets:
            label_names = {}
            df = params[dset]
            predicted_labels = pd.Series(params["prediction_" + dset])
            target_labels = df.get_labeled_data()["labels"]
            label_names.update(df.label_name)
            for i in pi:
                i.private_parameters = {
                    "target": target_labels.values.ravel(),
                    "prediction": predicted_labels.values.ravel(),
                    "classes": [i[1] for i in label_names.items()],
                    "data_file": params["input_data"],
                    "features_file": params[dset]
                }
                self.logger.info(f"Performance indicators {i.__class__.__name__} of results from {dset}")
                self.save_performance_indicators_results(f"{i.__class__.__name__}_{dset}", i, params)
        return results

    # def save_computed_performance_indicators(self, params: dict):
    #     pi = params["performance_indicators"]
    #
    #     # Save methods and results
    #     location = params["models_directory"]
    #     for i, p in enumerate(pi):
    #         cur_loc = join(location, f'{p}')
    #         self.logger.info(f"Saving performance indicator {p}")
    #         p.save(cur_loc)
    #
    #     return {}

    def save_performance_indicators_results(self, file_name, performance_indicator, params: dict):
        # Save methods and results
        location = params["results_directory"]
        cur_loc = join(location, file_name)
        self.logger.info(f"Saving performance indicator {performance_indicator}")
        performance_indicator.save_results(cur_loc)

        return {}

    def train_methods(self, params: dict):
        params["data_set_name"] = "training_set"

        # Train features extractors
        params.update(self.train_features(params))

        # Save trained features extractors
        params.update(self.save_trained_features_extractors(params))

        # Transform data
        params.update(self.transform_features(params))

        # Train machine learning method
        params.update(self.train_machine_learning_method(params))

        # Save ML method
        params.update(self.save_trained_ml_method(params))

        # Predict using the training set for training score
        params.update(self.predict_machine_learning_method(params))

        # Compute performance indicators
        params.update(self.compute_performance_indicators(params))

        # Save ml_method
        params.update(self.save_trained_ml_method(params))

        return {}

    def predict_methods(self, params: dict):
        params["data_set_name"] = "testing_set"

        # Load pretrained features
        params.update(self.load_trained_features_extractors(params))

        # Transform data
        params.update(self.transform_features(params))

        # Load pretrained model
        params.update(self.load_trained_machine_learning_method(params))

        # Predict using the training set for training score
        params.update(self.predict_machine_learning_method(params))

        # Compute performance indicators
        params.update(self.compute_performance_indicators(params))

        return {}
