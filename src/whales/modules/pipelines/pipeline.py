"""Package to parse a parameters file and execute what the user requests"""

import sys
import logging
import importlib
import json
from os.path import abspath, basename, dirname, join
from glob import glob
from multiprocessing import Process
from whales.modules.module import Module

functions_logger = logging.getLogger(__name__)


def get_available_features_extractors():
    """Return the available feature extraction classes.
    :return: dict whose keys are feature extractor method names and whose values are the
    method classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "features_extractors", "*.py")
    file_list = glob(r)

    file_list = [basename(f) for f in file_list]
    method_name = [f.split(".")[0]
                   for f in file_list if not f.startswith("__")]

    d = join(this_dir, "..", "..",  "..")
    d = abspath(d)
    sys.path.append(d)

    # Store those that contain a PipelineMethod named object (design requirement)
    m_names = []
    m_classes = []
    for n in method_name:
        try:
            m_classes.append(importlib.import_module(
                "whales.modules.features_extractors.{}".format(n)).PipelineMethod)
            m_names.append(n)
        except AttributeError:
            functions_logger.debug(f"Module {n} doesn't have a valid method")

    res = {}
    for b, c in zip(m_names, m_classes):
        res[b] = c

    return res


def get_available_pre_processing():
    """Return the available pre processing classes.
    :return: dict whose keys are pre processing method names and whose values are the
    method classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "pre_processing", "*.py")
    file_list = glob(r)

    file_list = [basename(f) for f in file_list]
    method_name = [f.split(".")[0]
                   for f in file_list if not f.startswith("__")]

    d = join(this_dir, "..", "..",  "..")
    d = abspath(d)
    sys.path.append(d)

    # Store those that contain a PipelineMethod named object (design requirement)
    m_names = []
    m_classes = []
    for n in method_name:
        try:
            m_classes.append(importlib.import_module(
                "whales.modules.pre_processing.{}".format(n)).PipelineMethod)
            m_names.append(n)
        except AttributeError:
            functions_logger.debug(f"Module {n} doesn't have a valid method")

    res = {}
    for b, c in zip(m_names, m_classes):
        res[b] = c

    return res


def get_available_performance_indicators():
    """Return the available performance indicators classes.
    :return: dict whose keys are performance indicators method names and whose values are the
    method classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "performance_indicators", "*.py")
    file_list = glob(r)
    file_list = [basename(f) for f in file_list]

    pi_name = [f.split(".")[0]
               for f in file_list if not f.startswith("__")]

    d = join(this_dir, "..", "..", "..")
    d = abspath(d)
    sys.path.append(d)

    # Store those that contain a PipelineMethod object
    pi_names = []
    pi_classes = []
    for n in pi_name:
        try:
            pi_classes.append(importlib.import_module(
                "whales.modules.performance_indicators.{}".format(n)).PipelineMethod)
            pi_names.append(n)
        except AttributeError:
            functions_logger.debug(f"Module {n} doesn't have a valid method'")

    res = dict()
    for a, b in zip(pi_names, pi_classes):
        res[a] = b

    return res


def get_available_unsupervised_methods():
    """Return the available unsupervised methods classes.
    :return: dict whose keys are unsupervised methods names and whose values are the
    methods classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "unsupervised", "*.py")
    file_list = glob(r)
    file_list = [basename(f) for f in file_list]

    cl_name = [f.split(".")[0]
               for f in file_list if not f.startswith("__")]

    d = join(this_dir, "..", "..", "..")
    d = abspath(d)
    sys.path.append(d)

    # Store those that contain a PipelineMethod object
    cl_names = []
    cl_classes = []
    for n in cl_name:
        try:
            cl_classes.append(importlib.import_module(
                "whales.modules.unsupervised.{}".format(n)).PipelineMethod)
            cl_names.append(n)
        except AttributeError:
            functions_logger.debug(f"Module {n} doesn't have a valid method'")

    res = dict()
    for a, b in zip(cl_names, cl_classes):
        res[a] = b

    return res


def get_available_supervised_methods():
    """Return the available supervised methods classes.
    :return: dict whose keys are supervised methods names and whose values are the
    methods classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "supervised", "*.py")
    file_list = glob(r)
    file_list = [basename(f) for f in file_list]

    cl_name = [f.split(".")[0]
               for f in file_list if not f.startswith("__")]

    d = join(this_dir, "..", "..", "..")
    d = abspath(d)
    sys.path.append(d)

    # Store those that contain a PipelineMethod object
    cl_names = []
    cl_classes = []
    for n in cl_name:
        try:
            cl_classes.append(importlib.import_module(
                "whales.modules.supervised.{}".format(n)).PipelineMethod)
            cl_names.append(n)
        except AttributeError:
            functions_logger.debug(f"Module {n} doesn't have a valid method'")

    res = dict()
    for a, b in zip(cl_names, cl_classes):
        res[a] = b

    return res


def get_available_semi_supervised_methods():
    """Return the available semi_supervised methods classes.
    :return: dict whose keys are semi_supervised methods names and whose values are the
    methods classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "semi_supervised", "*.py")
    file_list = glob(r)
    file_list = [basename(f) for f in file_list]

    cl_name = [f.split(".")[0]
               for f in file_list if not f.startswith("__")]

    d = join(this_dir, "..", "..", "..")
    d = abspath(d)
    sys.path.append(d)

    # Store those that contain a PipelineMethod object
    cl_names = []
    cl_classes = []
    for n in cl_name:
        try:
            cl_classes.append(importlib.import_module(
                "whales.modules.semi_supervised.{}".format(n)).PipelineMethod)
            cl_names.append(n)
        except AttributeError:
            functions_logger.debug(f"Module {n} doesn't have a valid method'")

    res = dict()
    for a, b in zip(cl_names, cl_classes):
        res[a] = b

    return res


def get_available_formatters():
    """Return the available formatter classes.
    :return: dict whose keys are formatters names and whose values are the
    methods classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "formatters", "*.py")
    file_list = glob(r)
    file_list = [basename(f) for f in file_list]

    fmt_name = [f.split(".")[0]
                for f in file_list if not f.startswith("__")]

    d = join(this_dir, "..", "..", "..")
    d = abspath(d)
    sys.path.append(d)

    # Store those that contain a PipelineMethod object
    fmt_names = []
    fmt_classes = []
    for n in fmt_name:
        try:
            fmt_classes.append(importlib.import_module(
                "whales.modules.formatters.{}".format(n)).PipelineFormatter)
            fmt_names.append(n)
        except AttributeError:
            functions_logger.debug(f"Module {n} doesn't have a valid method'")

    res = dict()
    for a, b in zip(fmt_names, fmt_classes):
        res[a] = b

    return res


def get_available_labels_formatters():
    """Return the available formatter classes.
    :return: dict whose keys are labels_formatters names and whose values are the
    methods classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "labels_formatters", "*.py")
    file_list = glob(r)
    file_list = [basename(f) for f in file_list]

    fmt_name = [f.split(".")[0]
                for f in file_list if not f.startswith("__")]

    d = join(this_dir, "..", "..", "..")
    d = abspath(d)
    sys.path.append(d)

    # Store those that contain a PipelineMethod object
    fmt_names = []
    fmt_classes = []
    for n in fmt_name:
        try:
            fmt_classes.append(importlib.import_module(
                "whales.modules.labels_formatters.{}".format(n)).PipelineFormatter)
            fmt_names.append(n)
        except AttributeError:
            functions_logger.debug(f"Module {n} doesn't have a valid method'")

    res = dict()
    for a, b in zip(fmt_names, fmt_classes):
        res[a] = b

    return res


def get_available_datafiles():
    """Return the available datafile classes.
    :return: dict whose keys are datafile names and whose values are the
    methods classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "data_files", "*.py")
    file_list = glob(r)
    file_list = [basename(f) for f in file_list]

    fmt_name = [f.split(".")[0]
                for f in file_list if not f.startswith("__")]

    d = join(this_dir, "..", "..", "..")
    d = abspath(d)
    sys.path.append(d)

    # Store those that contain a PipelineMethod object
    fmt_names = []
    fmt_classes = []
    for n in fmt_name:
        try:
            fmt_classes.append(importlib.import_module(
                "whales.modules.data_files.{}".format(n)).PipelineDatafile)
            fmt_names.append(n)
        except AttributeError:
            functions_logger.debug(f"Module {n} doesn't have a valid method'")

    res = dict()
    for a, b in zip(fmt_names, fmt_classes):
        res[a] = b

    return res


def get_available_pipeline_types():
    """Return the available pipelines types classes.
    :return: dict whose keys are pipelines types and whose values are the
    methods classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "pipelines", "*.py")
    file_list = glob(r)
    file_list = [basename(f) for f in file_list]

    fmt_name = [f.split(".")[0]
                for f in file_list if not f.startswith("__")]

    d = join(this_dir, "..", "..", "..")
    d = abspath(d)
    sys.path.append(d)

    # Store those that contain a PipelineType object
    fmt_names = []
    fmt_classes = []
    for n in fmt_name:
        try:
            fmt_classes.append(importlib.import_module(
                "whales.modules.pipelines.{}".format(n)).PipelineType)
            fmt_names.append(n)
        except AttributeError:
            functions_logger.debug(f"Module {n} doesn't have a valid method'")

    res = dict()
    for a, b in zip(fmt_names, fmt_classes):
        res[a] = b

    return res


def get_available_datasets():
    """Return the available data sets types classes.
    :return: dict whose keys are data sets and whose values are the
    methods classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "data_sets", "*.py")
    file_list = glob(r)
    file_list = [basename(f) for f in file_list]

    fmt_name = [f.split(".")[0]
                for f in file_list if not f.startswith("__")]

    d = join(this_dir, "..", "..", "..")
    d = abspath(d)
    sys.path.append(d)

    # Store those that contain a PipelineType object
    fmt_names = []
    fmt_classes = []
    for n in fmt_name:
        try:
            fmt_classes.append(importlib.import_module(
                "whales.modules.data_sets.{}".format(n)).PipelineDataSet)
            fmt_names.append(n)
        except AttributeError:
            functions_logger.debug(f"Module {n} doesn't have a valid method'")

    res = dict()
    for a, b in zip(fmt_names, fmt_classes):
        res[a] = b

    return res


class Pipeline(Module):
    """Module that parses JSON formatted parameter set, sets up the pipelines and allows launching it.
    Actual pipelines initialization or instructions are implemented in children classes."""
    def __init__(self, logger=logging.getLogger(__name__)):
        super(Pipeline, self).__init__(logger)
        self.process = None
        self.description = "Generic Pipeline"
        self.instructions_series = []
        self.results = {}

        # Default parameters
        self.parameters = {
            "necessary_parameters": {},
            "optional_parameters": {},
            "expected_input_parameters": {
                "file_name": str,
                "data_file": str,
                "formatter": str,
            },
            "expected_labels_parameters": {
                "labels_file": str,
                "labels_formatter": str,
            }
        }

    def start(self):
        """Run the instructions series"""
        self.logger.debug(f"Pipeline started")
        self.process = Process(target=self.instructions)
        self.process.start()

    def load_parameters(self, parameters_file: str):
        dictionary = json.loads(parameters_file)
        self.parameters = self.parse_parameters(dictionary)
        self.logger.debug("Parameters set")

    def instructions(self):
        pass

    def add_instruction(self, instruction_type, instruction_parameters):
        """Adds instruction to the last execution place"""
        self.instructions_series.append((instruction_type, instruction_parameters))

    def next_instruction(self):
        """Returns the next instruction"""
        if len(self.instructions_series) == 0:
            instruction = None
        else:
            instruction = self.instructions_series.pop(0)
        return instruction

    def parse_parameters(self, parameters_dict):
        self.logger.debug("Parsing pipeline parameters")

        # Parse for necessary parameters
        for key in self.parameters["necessary_parameters"]:
            if key not in parameters_dict:
                self.logger.error(f"Parameters dict does not include necessary parameter: {key}")
                raise ValueError(f"Parameters dict does not include necessary parameter: {key}")

        expected_parameters = {**self.parameters["necessary_parameters"], **self.parameters["optional_parameters"]}
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

        self._parse_parameters_structure(parameters_dict)
        self._parse_input_data(parameters_dict["input_data"])
        self._parse_input_labels(parameters_dict["input_labels"])

        return parameters_dict

    def _parse_parameters_structure(self, current_elem):
        """
        1. Make sure that sub_dict's lists have only dictionaries inside.
        :param current_elem:
        :return:
        """
        if type(current_elem) is list:
            for next_elem in current_elem:
                if type(next_elem) is dict:
                    self._parse_parameters_structure(next_elem)
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
                    self._parse_parameters_structure(val)

    def _parse_input_data(self, input_data):
        expected_parameters = self.parameters["expected_input_parameters"]
        for elem in input_data:
            if type(elem) is not dict:
                raise ValueError("Data should be specified in a dict")
            for p in expected_parameters:
                if p not in elem:
                    raise ValueError(f"Parameter {p} missing from input files specification")
                elif type(elem[p]) is not expected_parameters[p]:
                    raise ValueError((
                        f"Incorrect type for {p} in input files specification. It should be a ",
                        f"{expected_parameters[p]} and it is actually a {type(elem[p])}"
                    ))

    def _parse_input_labels(self, input_labels):
        expected_parameters = self.parameters["expected_labels_parameters"]
        for elem in input_labels:
            if type(elem) is not dict:
                raise ValueError("Labels should be specified in a dict")
            for p in expected_parameters:
                if p not in elem:
                    raise ValueError(f"Parameter {p} missing from input labels specification")
                elif type(elem[p]) is not expected_parameters[p]:
                    raise ValueError((
                        f"Incorrect type for {p} in input labels specification. It should be a ",
                        f"{expected_parameters[p]} and it is actually a {type(elem[p])}"
                    ))

    def load_input_data(self):
        """Add instructions to load the input data. Also, use glob where stars are detected"""
        real_input_data = []
        for elem in self.parameters["input_data"]:
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
        data_set_type = self.parameters.get("data_set_type", "files_fold")
        self.add_instruction("build_data_set",
                             {
                                 "input_data": real_input_data,
                                 "data_set_type": data_set_type
                             })

    def load_labels(self):
        """Add instructions to load the labels. Also, use glob where stars are detected"""
        real_labels = []
        for elem in self.parameters["input_data"]:
            if "labels_file" in elem:
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
        self.add_instruction("set_labels", {"input_labels": real_labels})

    def get_commands(self):
        """Return a dict with the available instructions to execute"""
        def build_data_set(params):
            available_data_sets = get_available_datasets()
            available_data_files = get_available_datafiles()
            available_formatters = get_available_formatters()
            ds = available_data_sets[params["data_set_type"]]()
            for elem in params["input_data"]:
                file_name = elem["file_name"]
                data_file_name = elem["data_file"]
                formatter_name = elem["formatter"]
                df = available_data_files[data_file_name]()
                fmt = available_formatters[formatter_name]()
                df.load_data(file_name=file_name, formatter=fmt)
                ds.add_datafile(df)
            return {"data_set": ds}

        def set_labels(params):
            results = params.get("results", {})
            data_set = results.get("data_set")
            pass

        return {
            "build_data_set": build_data_set,
            #"set_labels": set_labels
        }