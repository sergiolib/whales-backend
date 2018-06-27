"""Computing heavy instructions for generating step results throughout the pipeline"""

import logging

from whales.modules.data_files.feature import FeatureDataFile
from whales.modules.pipelines.getters import get_available_data_sets, get_available_datafiles, get_available_formatters, \
    get_available_labels_formatters


class InstructionSet:
    def __init__(self, logger=None):
        self.logger = logger
        if self.logger is None:
            logging.getLogger(self.__class__.__name__)


class SupervisedWhalesInstructionSet(InstructionSet):
    def build_data_file(self, params:  dict):
        """"""
        available_data_files = get_available_datafiles()
        available_formatters = get_available_formatters()

        # Load every small input data file and concatenate all into the big data file
        dfs = []
        for elem in params["input_data"]:
            file_name = elem["file_name"]
            data_file_name = elem["data_file"]
            formatter_name = elem["formatter"]
            df = available_data_files[data_file_name]()
            fmt = available_formatters[formatter_name]()
            df.load_data(file_name=file_name, formatter=fmt)
            dfs.append(df)
        big_df = dfs[0].concatenate(dfs)

        return {"input_data": big_df}

    def set_labels(self, params:  dict):
        labels_params = params["input_labels"]
        input_data = params["input_data"]

        lf = get_available_labels_formatters()

        for p in labels_params:
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
        transformed_training_set = params["transformed_training_set"]
        y = transformed_training_set.data["labels"].values.astype(int)
        x = transformed_training_set.data.drop("labels", axis=1).values
        ml_method.fit(x, y)
        return {}

    def train_performance_indicators(self, params: dict):
        pi = params["performance_indicators"]
        current_training_set = params["current_training_set"]
        x = current_training_set.data.drop("labels", axis=1)
        for p in pi:
            p.fit(x.values)
        return {}

    def train_features(self, params: dict):
        feat = params["features_extractors"]
        current_training_set = params["current_training_set"]
        x = current_training_set.data.drop("labels", axis=1)
        for f in feat:
            f.fit(x.values)
        return {}

    def transform_features(self, params: dict):
        feat = params["features_extractors"]
        current_set = {}
        transformed_set = {}
        labels_set = {}
        for s in ["training", "testing", "validation"]:
            current_set[s] = params[f"current_{s}_set"]
            x = current_set[s].data.drop("labels", axis=1).values.astype(float)
            transformed_set[s] = []
            for f in feat:
                res = f.transform(x)
                transformed_set[s].append(res)

            transformed_set[s] = FeatureDataFile().concatenate(transformed_set[s])
            labels_set[s] = current_set[s].data["labels"].astype(int)
            transformed_set[s].data["labels"] = labels_set[s]
            transformed_set[s].data = transformed_set[s].data.dropna(axis="rows")
            labels_set[s] = transformed_set[s].data["labels"]

        return {
            "transformed_training_set": transformed_set["training"],
            "transformed_testing_set": transformed_set["testing"],
            "transformed_validation_set": transformed_set["validation"],
            "labels_training": labels_set["training"].values.ravel(),
            "labels_testing": labels_set["testing"].values.ravel(),
            "labels_validation": labels_set["validation"].values.ravel(),
        }

    def transform_pre_processing(self, params: dict):
        pre_processing_methods = params["pre_processing_methods"]
        input_data = params["input_data"]
        data = input_data
        for pp in pre_processing_methods:
            data = pp.transform(data)
        output_data = data
        return {"input_data": output_data}

    def predict_machine_learning_method(self, params: dict):
        ml_method = params["ml_method"]
        results = {}
        for dset in ["training", "testing", "validation"]:
            df = params[f"transformed_{dset}_set"]
            x = df.data.drop("labels", axis=1).values
            prediction = ml_method.predict(x)
            results[f"prediction_{dset}"] = prediction
        return {**results}

    def compute_performance_indicators(self, params: dict):
        pi = params["performance_indicators"]
        results = {}
        for dset in ["testing", "validation"]:
            ns = params["number_of_sets"]
            predicted_labels = []
            target_labels = []
            for i in range(ns):
                predicted_labels += params[f"{i + 1}/{ns}"][f"prediction_{dset}"].tolist()
                target_labels += params[f"{i + 1}/{ns}"][f"labels_{dset}"].tolist()
            for i in pi:
                i.parameters = {
                    "target": target_labels,
                    "prediction": predicted_labels,
                }
                res = i.compute()
                results[f"pi_{i.__class__.__name__}_{dset}"] = res
        return {**results}

    def build_data_set(self, params: dict):
        available_data_sets = get_available_data_sets()
        method = params["ds_options"]["method"]
        ds_cls = available_data_sets[method]
        ds = ds_cls()
        data_file = params["input_data"]
        ds.add_data_file(data_file)
        data_generator = ds.get_data_frames()
        number_of_sets = ds.iterations
        return {"data_generator": data_generator, "number_of_sets": number_of_sets}

    def train_execute_methods(self, params: dict):
        data_generator = params["data_generator"]

        number_of_sets = params["number_of_sets"]

        # All iterations results dictionary
        results = {}

        # Iterate on sets
        for iteration, (current_training, current_testing, current_validation) in enumerate(data_generator):
            # Set current sets
            params["current_training_set"] = current_training
            params["current_testing_set"] = current_testing
            params["current_validation_set"] = current_validation

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
            results[f"{iteration + 1}/{number_of_sets}"] = params

        # Compute performance indicators
        params.update(self.compute_performance_indicators({**results, **params}))

        return results
