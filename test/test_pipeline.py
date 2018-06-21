import sys, os

import pytest
from glob import glob

from utilities import get_5_file_names, get_labeled
from whales.modules.pipelines.whale_detector import WhaleDetector

myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../')

from whales.modules.pipelines.pipeline import get_available_features_extractors
from whales.modules.pipelines.pipeline import get_available_pre_processing
from whales.modules.pipelines.pipeline import get_available_pipeline_types
from whales.modules.pipelines.pipeline import get_available_formatters
from whales.modules.pipelines.pipeline import get_available_labels_formatters
from whales.modules.pipelines.pipeline import get_available_datafiles
from whales.modules.pipelines.pipeline import get_available_performance_indicators
from whales.modules.pipelines.pipeline import get_available_unsupervised_methods
from whales.modules.pipelines.pipeline import get_available_supervised_methods
from whales.modules.pipelines.pipeline import get_available_semi_supervised_methods
from whales.modules.pipelines.pipeline import get_available_datasets


def test_get_available_feature_extractors():
    """Test that get_available_feature_extractors function works correctly"""
    fe = get_available_features_extractors()
    assert "identity" in fe


def test_get_available_pre_processing():
    """Test that get_available_feature_extractors function works correctly"""
    fe = get_available_pre_processing()
    assert "scale" in fe


def test_get_available_performance_indicators():
    """Test that get_available_performance_indicators function works correctly"""
    pi = get_available_performance_indicators()
    assert "accuracy" in pi


def test_get_available_unsupervised_methods():
    """Test that get_available_unsupervised_methods function works correctly"""
    cl = get_available_unsupervised_methods()
    # assert "kmeans" in cl


def test_get_available_supervised_methods():
    """Test that get_available_supervised_methods function works correctly"""
    cl = get_available_supervised_methods()
    # assert "logistic_regression" in cl


def test_get_available_semi_supervised_methods():
    """Test that get_available_semi_supervised_methods function works correctly"""
    cl = get_available_semi_supervised_methods()
    # assert "darr" in cl


def test_get_available_formatters():
    """Test that get_available_formatters function works correctly"""
    pi = get_available_formatters()
    assert "hdf5" in pi


def test_get_available_labels_formatters():
    """Test that get_available_labels_formatters function works correctly"""
    lf = get_available_labels_formatters()
    assert "csv" in lf


def test_get_available_datafiles():
    """Test that get_available_datafiles function works correctly"""
    cl = get_available_datafiles()
    assert "audio" in cl


def test_get_available_pipeline_types():
    """Test that get_available_pipeline_types function works correctly"""
    cl = get_available_pipeline_types()
    assert "whale_detector" in cl


def test_get_available_datasets():
    """Test that get_available_datasets function works correctly"""
    ds = get_available_datasets()
    assert "files_fold" in ds


def test_load_parameters():
    """Test that loading set of parameters works the way it is expected"""
    p = WhaleDetector()

    demo_parameters = """
{
    "pipeline_type": "whale_detector",
    "input_data": [
        {
            "file_name": "demo_data.h5",
            "data_file": "time_series",
            "formatter": "hdf5"
        }
    ],
    "input_labels": [{
        "labels_file": "demo_labels.txt",
        "labels_formatter": "txt"
    }],
    "pre_processing": [
        {
            "method": "scale",
            "parameters": {}
        }
    ],
    "features_extractors": [
        {
            "method": "identity",
            "parameters": {}
        }
    ],
    "machine_learning": {
        "type": "unsupervised",
        "method": "kmeans",
        "parameters": {}
    },
    "performance_indicators": [
        {
            "method": "accuracy",
            "parameters": {}
        }
    ],
    "output_directory": "./demo",
    "data_set_type": {
        "method": "files_fold"
    }
}
    """

    p.load_parameters(demo_parameters)
    assert type(p.parameters["machine_learning"]) is dict
    assert p.parameters["performance_indicators"][0]["method"] == "accuracy"


def test_missing_necessary_parameters():
    p = WhaleDetector()

    missing_parameters = """
{
    "input_data": [
        {
            "file_name": "demo_data.h5",
            "data_file": "time_series",
            "formatter": "hdf5"
        }
    ],
    "output_directory": "./demo"
}
    """

    with pytest.raises(ValueError):
        p.load_parameters(missing_parameters)


def test_extra_parameters():
    p = WhaleDetector()

    extra_parameters = """
{
    "pipeline_type": "whale_detector",
    "input_data": [
        {
            "file_name": "demo_data.h5",
            "data_file": "time_series",
            "formatter": "hdf5"
        }
    ],
    "output_directory": "./demo",
    "temperature": 100.0
}
    """

    with pytest.raises(ValueError):
        p.load_parameters(extra_parameters)


def test_wrong_parameters():
    p = WhaleDetector()

    wrong_format_parameters = """
{
    "pipeline_type": "whale_detector",
    "output_directory": "./demo",
    "input_data": [
        "demo_data.hdf5",
        "demo_data.aiff"
    ],
}
    """

    with pytest.raises(ValueError):
        p.load_parameters(wrong_format_parameters)


def test_whales_pipeline():
    _ = get_5_file_names()
    p = WhaleDetector()
    [os.remove(file) for file in glob("*.aif")]
    labeled_fns = get_labeled()
    parameters = """{
        "pipeline_type": "whale_detector",
        "input_data": [
            {
                "file_name": "*.aif",
                "data_file": "audio",
                "formatter": "aif"
            }
        ],
        "input_labels": [{
            "labels_file": "*.csv",
            "labels_formatter": "csv"
        }],
        "output_directory": "./demo"
    }"""
    p.load_parameters(parameters)
    p.initialize()
    p.start()
    ds = p.results["data_set"]
    tr = ds.get_training()
    for t in tr:
        return