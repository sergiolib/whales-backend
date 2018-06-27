import importlib
import sys
from glob import glob
from os.path import abspath, dirname, basename, join

import logging
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


def get_available_data_files():
    """Return the available data_file classes.
    :return: dict whose keys are data_file names and whose values are the
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
                "whales.modules.data_files.{}".format(n)).PipelineDataFile)
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


def get_available_data_sets():
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
