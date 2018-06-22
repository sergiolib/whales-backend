import logging


class Parser:
    def __init__(self, pipeline, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)

        self.pipeline = pipeline


class PipelineParser(Parser):
    def parse(self, parameters_dict):
        self.logger.debug("Parsing pipeline parameters")

        # Parse for necessary parameters
        for key in self.pipeline.parameters["necessary_parameters"]:
            if key not in parameters_dict:
                self.logger.error(f"Parameters dict does not include necessary parameter: {key}")
                raise ValueError(f"Parameters dict does not include necessary parameter: {key}")

        expected_parameters = {**self.pipeline.parameters["necessary_parameters"], **self.pipeline.parameters["optional_parameters"]}
        for key in parameters_dict:
            if key in expected_parameters:
                actual_type = type(parameters_dict[key])
                expected_type = expected_parameters[key]
                if actual_type is not expected_type:
                    self.logger.error(
                        f"Expected type is {expected_type} and obtained type is {actual_type} for parameter {key}"
                    )
                    raise ValueError(
                        f"Expected type is {expected_type} and obtained type is {actual_type} for parameter {key}"
                    )
            else:
                self.logger.error(f"Parameters dict included unexpected key {key}")
                raise ValueError(f"Parameters dict included unexpected key {key}")

        self.parse_parameters_structure(parameters_dict)
        self.parse_input_data(parameters_dict["input_data"])
        self.parse_input_labels(parameters_dict["input_labels"])
        self.parse_features_extractors(parameters_dict["features_extractors"])

        return parameters_dict

    def parse_parameters_structure(self, current_elem):
        """
        1. Make sure that sub_dict's lists have only dictionaries inside.
        :param current_elem:
        :return:
        """
        if type(current_elem) is list:
            for next_elem in current_elem:
                if type(next_elem) is dict:
                    self.parse_parameters_structure(next_elem)
                else:
                    raise ValueError(
                        (
                            f"Lists should not have {type(next_elem)} elements in the parameters ",
                            f"dictionary, only dicts"
                        )
                    )
        elif type(current_elem) is dict:
            for _, val in current_elem.items():
                if type(val) is dict:
                    self.parse_parameters_structure(val)

    def parse_input_data(self, input_data):
        expected_parameters = self.pipeline.parameters["expected_input_parameters"]
        for elem in input_data:
            if type(elem) is not dict:
                raise ValueError("Data should be specified in a dict")
            for k in elem:
                if k not in expected_parameters:
                    raise ValueError(f"Parameter {p} was not expected in input files specification")
            for p in expected_parameters:
                if p not in elem:
                    raise ValueError(f"Parameter {p} missing from input files specification")
                elif type(elem[p]) is not expected_parameters[p]:
                    raise ValueError((
                        f"Incorrect type for {p} in input files specification. It should be a ",
                        f"{expected_parameters[p]} and it is actually a {type(elem[p])}"
                    ))

    def parse_input_labels(self, input_labels):
        expected_parameters = self.pipeline.parameters["expected_labels_parameters"]
        for elem in input_labels:
            if type(elem) is not dict:
                raise ValueError("Labels should be specified in a dict")
            for k in elem:
                if k not in expected_parameters:
                    raise ValueError(f"Parameter {p} was not expected in input labels specification")
            for p in expected_parameters:
                if p not in elem:
                    raise ValueError(f"Parameter {p} missing from input labels specification")
                elif type(elem[p]) is not expected_parameters[p]:
                    raise ValueError((
                        f"Incorrect type for {p} in input labels specification. It should be a ",
                        f"{expected_parameters[p]} and it is actually a {type(elem[p])}"
                    ))

    def parse_features_extractors(self, features_extractors):
        expected_parameters = self.pipeline.parameters["expected_features_parameters"]
        for elem in features_extractors:
            if type(elem) is not dict:
                raise ValueError("Features extractors should be specified each one in a dict")
            for k in elem:
                if k not in expected_parameters:
                    raise ValueError(f"Parameter {p} was not expected in feature extractors specification")
            for p in expected_parameters:
                if p not in elem:
                    raise ValueError(f"Parameter {p} missing from features extractors specification")
                elif type(elem[p]) is not expected_parameters[p]:
                    raise ValueError((
                        f"Incorrect type for {p} in features extractors specification. It should be a ",
                        f"{expected_parameters[p]} and it is actually a {type(elem[p])}"
                    ))
