"""Package to parse a parameters file and execute what the user requests"""

import sys
import logging
import importlib
from os.path import abspath, basename, dirname, join
from glob import glob

logger = logging.getLogger(__name__)


def get_available_feature_extractors():
    """Return the available feature extraction classes.
    :return: dict whose keys are feature extractor method names and whose values are the
    method classes"""
    this_dir = abspath(dirname(__file__))
    r = join(this_dir, "..", "feature_extraction", "*.py")
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
                "whales.modules.feature_extraction.{}".format(n)).PipelineMethod)
            m_names.append(n)
        except AttributeError:
            logger.debug("Module {} doesn't have a valid method".format(n))

    res = {}
    for b, c in zip(m_names, m_classes):
        res[b] = c

    return res
