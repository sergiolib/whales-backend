import logging


class Parser:
    def __init__(self, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)


class WhalesPipelineParser(Parser):
    def __init__(self, logger=None):
        super(WhalesPipelineParser, self).__init__(logger)
        self.expected_fields_types = {
            "output_directory": str,
            "pipeline_type": str,
            "input_data": [({
                                "file_name": str,
                                "data_file": str,
                                "formatter": str
                            }, "optional")],
            "input_labels": [({"labels_file": str, "labels_formatter": str}, "optional")],
            "pre_processing": [({"method": str, "parameters": (dict, "optional")}, "optional")],
            "features_extractors": [({"method": str, "parameters": (dict, "optional")}, "optional")],
            "performance_indicators": [({"method": str, "parameters": (dict, "optional")}, "optional")],
            "machine_learning": {"type": str, "method": str, "parameters": (dict, "optional")},
            "data_set_type": {"method": str, "parameters": (dict, "optional")},
            "active": (bool, "optional"),
            "verbose": (bool, "optional"),
            "seed": (int, "optional"),
        }

    def parse(self, parameters_dict):
        self.logger.debug("Parsing pipeline parameters")
        self.parse_field(parameters_dict, (self.expected_fields_types, "necessary"))
        return parameters_dict

    def parse_field(self, parameters, expected_types):
        field_type = type(parameters)
        if type(expected_types) is tuple:
            expected_type = expected_types[0] if type(expected_types[0]) is type else type(expected_types[0])
        else:
            expected_type = expected_types if type(expected_types) is type else type(expected_types)
        expected_types = expected_types[0] if type(expected_types) is tuple else expected_types
        if field_type is not expected_type:
            raise UnexpectedTypeError(field_type, expected_type)
        if field_type is dict:
            for key in parameters:
                try:
                    self.parse_field(parameters[key], expected_types[key])
                except KeyError:
                    raise ValueError(f"Parameter {key} was not expected in {parameters} specification")
                except UnexpectedTypeError as e:
                    raise ValueError(f"Parameter {key} has unexpected type: obtained {e.obtained} ",
                                     f"while expecting {e.expected}")
            expected_types_not_in_parameters = set(expected_types.keys()) - set(parameters.keys())
            for key in expected_types_not_in_parameters:
                if type(expected_types[key]) is tuple and expected_types[key][1] == "optional":
                    pass
                else:
                    raise ValueError(f"Expecting parameter {key} in object {parameters} but couldn't find it")
        if field_type is list:
            for elem, exp in zip(parameters, expected_types):
                self.parse_field(elem, exp)


class UnexpectedTypeError(BaseException):
    def __init__(self, obtained, expected):
        super(UnexpectedTypeError, self).__init__()
        self.obtained = obtained
        self.expected = expected
