import logging


class Parser:
    def __init__(self, logger=None):
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)

    def parse(self, parameters_dict):
        self.parse_field(parameters_dict, (self.expected_fields_types, "necessary"))
        self.logger.info("Parsed pipeline parameters")
        return parameters_dict

    def parse_field(self, parameters, expected_types):
        """Parses the dictionary of parameters comparing it to the one of expected_types.
        Discrepancies are show to the user as informative exceptions or logs."""
        field_type = type(parameters)
        if type(expected_types) is tuple:
            expected_type = expected_types[0] if type(expected_types[0]) is type else type(expected_types[0])
        else:
            expected_type = expected_types if type(expected_types) is type else type(expected_types)
        expected_types = expected_types[0] if type(expected_types) is tuple else expected_types
        if field_type is not expected_type:
            raise UnexpectedTypeError(field_type, expected_type)
        if field_type is dict:
            if type(expected_types) is type:
                return  # Happens when no further search is needed in the tree (i.e. search has reached dict object)
            for key in parameters:
                try:
                    self.parse_field(parameters[key], expected_types[key])
                except KeyError:
                    raise UnexpectedParameterError(key,
                                                   f"Parameter {key} with value {parameters[key]} "
                                                   f"was not expected in specification")
                except UnexpectedTypeError as e:
                    raise UnexpectedTypeError(e.obtained, e.expected, (f"Parameter {key} has unexpected type: ",
                                                                       f"obtained {e.obtained} ",
                                                                       f"while expecting {e.expected}"))
                except NecessaryParameterAbsentError as e:
                    raise NecessaryParameterAbsentError(e.expected, f"Expecting parameter {e.expected} in "
                                                                    f"{key} but couldn't find it")
            expected_types_keys = set(expected_types.keys())
            parameters_keys = set(parameters.keys())
            expected_types_not_in_parameters = expected_types_keys - parameters_keys
            for key in expected_types_not_in_parameters:
                if type(expected_types[key]) is tuple and expected_types[key][1] == "optional":
                    pass
                else:
                    raise NecessaryParameterAbsentError(key)
        if field_type is list:
            for elem, exp in zip(parameters, expected_types):
                if type(exp) is type:
                    return  # Happens when no further search is needed in the tree (i.e. search has reached dict object)
                self.parse_field(elem, exp)


class UnexpectedTypeError(Exception):
    def __init__(self, obtained, expected, message=None):
        super().__init__(message)
        self.obtained = obtained
        self.expected = expected


class NecessaryParameterAbsentError(Exception):
    def __init__(self, expected, message=None):
        super().__init__(message)
        self.expected = expected


class UnexpectedParameterError(Exception):
    def __init__(self, parameter_name, message=None):
        super().__init__(message)
        self.parameter_name = parameter_name


class WhalesPipelineParser(Parser):
    def __init__(self, logger=None):
        super().__init__(logger)
        self.expected_fields_types = {  # Ground truth for a good pipeline configuration
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
