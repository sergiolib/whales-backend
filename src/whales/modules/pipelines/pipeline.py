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


def get_available_clustering_methods():
    """Return the available clustering methods classes.
    :return: dict whose keys are clustering methods names and whose values are the
    methods classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "clustering", "*.py")
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
                "whales.modules.clustering.{}".format(n)).PipelineMethod)
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


def get_available_datafiles():
    """Return the available datafile classes.
    :return: dict whose keys are datafile names and whose values are the
    methods classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "datafiles", "*.py")
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
                "whales.modules.datafiles.{}".format(n)).PipelineDatafile)
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
    r = join(this_dir, "..", "datasets", "*.py")
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
                "whales.modules.datasets.{}".format(n)).PipelineDataSet)
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
        self.id = "Generic Pipeline"
        self.instructions_series = []

        # Default parameters
        self.parameters = {}

    def start(self):
        """Run the instructions series"""
        self.logger.debug(f"Pipeline {self.id} started")
        self.process = Process(target=self.instructions)
        self.process.start()

    def load_parameters(self, parameters: str):
        self.parameters = json.loads(parameters)
        self.logger.debug("Parameters set")

    def initialize(self):
        """Use the attribute parameters and set the pipelines instructions set"""
        raise NotImplementedError

    def instructions(self):
        for method in self.instructions_series:
            self.logger.debug(f"Running method {method.__name__}")
            method()
